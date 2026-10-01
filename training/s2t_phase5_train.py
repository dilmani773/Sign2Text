"""
Sign2Text — Phase 5: Training
=============================

What it does
    1. Builds train / val datasets, cleans them, preloads into RAM
    2. Trains Sign2Text (pose encoder + T5) with AdamW, warmup + cosine LR,
       label smoothing, gradient clipping, fp32
    3. After each epoch: val loss + val BLEU-4 (greedy) + 3 example outputs
    4. Saves best.pt (by val BLEU), last.pt (for resume), history.csv
    5. Stops early if BLEU stops improving, or before Kaggle's time limit
    6. At the end: test BLEU with beam search, on the cleaned test set and on
       the full test set (the full one is what papers report)

Outputs in OUT_DIR (/kaggle/working):
    best.pt, last.pt, history.csv, test_predictions.csv,
    train_index.csv, val_index.csv, test_index.csv,
    train_pre.npz, val_pre.npz            (caches, reused next time)

Resume
    Set CONFIG['resume'] to a last.pt path. Without it, the script only
    resumes from last.pt in OUT_DIR (same session), never from an old run
    found in /kaggle/input, since that may be a different model.

Run history
    Run 1 (baseline): d=256, 4 layers, no BoW, lr_dec 1e-4
                      → val BLEU 2.03, test BLEU 1.67
    Run 2 (this):     bigger encoder, BoW helper loss, decoder frozen for
                      the first epochs, lower T5 LR, no-repeat generation

Run locally (tiny model, fake data, ~1 min on CPU):
    python s2t_phase5_train.py --smoke
"""

import os
import sys
import csv
import glob
import math
import time
import shutil
import logging
import argparse
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from s2t_phase1_dataset import How2SignS2T, Collator, clean_index
from s2t_phase34_model import Sign2Text, build_model, T5_NAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s',
                    datefmt='%H:%M:%S')
for noisy in ('httpx', 'httpcore', 'huggingface_hub', 'urllib3'):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger('train')


CONFIG = dict(
    data_root        = '/kaggle/input/datasets/psewmuthu/how2sign-holistic/how2sign_holistic_features',
    out_dir          = '/kaggle/working',
    t5_name          = T5_NAME,
    encoder          = dict(d=384, layers=6, heads=6, ff=1536, dropout=0.15),
    aux_weight       = 0.5,        # bag-of-words helper loss
    freeze_decoder_epochs = 3,     # encoder learns alone first

    batch_size       = 32,
    epochs           = 40,
    patience         = 8,          # epochs without BLEU gain before stopping
    lr_encoder       = 5e-4,
    lr_decoder       = 5e-5,
    weight_decay     = 0.01,
    warmup_steps     = 1500,
    min_lr_ratio     = 0.05,       # cosine ends at 5% of peak LR
    grad_clip        = 1.0,
    label_smoothing  = 0.1,

    max_frames       = 256,
    stride           = 2,
    max_text_len     = 64,

    num_beams_eval   = 1,          # greedy during training (fast)
    num_beams_test   = 4,
    max_hours        = 11.0,       # Kaggle limit is 12 h; leave time to save + test
    num_workers      = 2,
    seed             = 42,
    resume           = None,
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_file(name: str, out_dir: str) -> Optional[str]:
    """Look in out_dir first, then anywhere under /kaggle/input."""
    p = os.path.join(out_dir, name)
    if os.path.exists(p):
        return p
    hits = glob.glob(f'/kaggle/input/**/{name}', recursive=True)
    return hits[0] if hits else None


def split_paths(root: str, split: str):
    return (os.path.join(root, 'metadata', f'how2sign_realigned_{split}.csv'),
            os.path.join(root, split, 'frontal'))


def make_dataset(cfg: dict, split: str, augment: bool, clean: bool = True,
                 preload: bool = True) -> How2SignS2T:
    csv_path, npy_dir = split_paths(cfg['data_root'], split)
    out = cfg['out_dir']

    idx_name = f'{split}_index.csv'
    found = find_file(idx_name, out)
    if found and found != os.path.join(out, idx_name):
        shutil.copy(found, os.path.join(out, idx_name))

    ds = How2SignS2T(csv_path, npy_dir, max_frames=cfg['max_frames'], stride=cfg['stride'],
                     augment=augment, cache_path=os.path.join(out, idx_name))
    if clean:
        clean_index(ds)
    if preload:
        pre_name = f'{split}_pre.npz'
        found = find_file(pre_name, out)
        ds.preload(found or os.path.join(out, pre_name))
    return ds


def make_loader(ds, tok, cfg, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=cfg['batch_size'], shuffle=shuffle, drop_last=shuffle,
                      num_workers=cfg['num_workers'], collate_fn=Collator(tok, cfg['max_text_len']),
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=cfg['num_workers'] > 0)


def lr_lambda(warmup: int, total: int, min_ratio: float):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    return f


def bleu4(hyps: List[str], refs: List[str]) -> float:
    try:
        import sacrebleu
        return sacrebleu.corpus_bleu(hyps, [refs]).score
    except ImportError:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        return 100 * corpus_bleu([[r.lower().split()] for r in refs],
                                 [h.lower().split() for h in hyps],
                                 smoothing_function=SmoothingFunction().method1)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, tok, device, num_beams: int, max_new_tokens: int,
             label_smoothing: float = 0.0) -> Dict:
    model.eval()
    losses, hyps, refs, sids = [], [], [], []
    for b in loader:
        f, m = b['features'].to(device), b['feature_mask'].to(device)
        model(f, m, b['labels'].to(device), label_smoothing)
        losses.append(model.parts['ce'])                 # CE only, comparable across runs
        hyps += model.translate(f, m, tok, num_beams=num_beams, max_new_tokens=max_new_tokens)
        refs += b['texts']
        sids += b['sids']
    return {'loss': float(np.mean(losses)), 'bleu': bleu4(hyps, refs),
            'hyps': hyps, 'refs': refs, 'sids': sids}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def save_ckpt(path, model, opt, sched, epoch, best_bleu, bad_epochs, cfg) -> None:
    tmp = path + '.tmp'
    torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                'sched': sched.state_dict(), 'epoch': epoch, 'best_bleu': best_bleu,
                'bad_epochs': bad_epochs, 'cfg': cfg}, tmp)
    os.replace(tmp, path)                       # never leaves a half-written file


def train(cfg: dict, model=None, tok=None, train_ds=None, val_ds=None) -> Dict:
    t0 = time.time()
    set_seed(cfg['seed'])
    os.makedirs(cfg['out_dir'], exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == 'cuda' else ''))

    if model is None:
        model, tok = build_model(cfg['t5_name'], aux_weight=cfg['aux_weight'], **cfg['encoder'])
    log.info(f"Params: {model.count_params()}")
    model.to(device)
    if train_ds is None:
        train_ds = make_dataset(cfg, 'train', augment=True)
    if val_ds is None:
        val_ds = make_dataset(cfg, 'val', augment=False)
    train_dl = make_loader(train_ds, tok, cfg, shuffle=True)
    val_dl   = make_loader(val_ds, tok, cfg, shuffle=False)

    opt   = torch.optim.AdamW(model.param_groups(cfg['lr_encoder'], cfg['lr_decoder'],
                                                 cfg['weight_decay']))
    total = cfg['epochs'] * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(cfg['warmup_steps'], total, cfg['min_lr_ratio']))

    start_epoch, best_bleu, bad_epochs = 1, -1.0, 0
    here = os.path.join(cfg['out_dir'], 'last.pt')
    resume = cfg['resume'] or (here if os.path.exists(here) else None)
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start_epoch, best_bleu, bad_epochs = ck['epoch'] + 1, ck['best_bleu'], ck['bad_epochs']
        best_src = find_file('best.pt', cfg['out_dir'])
        if best_src and best_src != os.path.join(cfg['out_dir'], 'best.pt'):
            shutil.copy(best_src, os.path.join(cfg['out_dir'], 'best.pt'))
        log.info(f"Resumed from {resume}: epoch {ck['epoch']}, best BLEU {best_bleu:.2f}")

    hist_path = os.path.join(cfg['out_dir'], 'history.csv')
    hist_src  = find_file('history.csv', cfg['out_dir'])
    if resume and hist_src and hist_src != hist_path:
        shutil.copy(hist_src, hist_path)
    if not (resume and os.path.exists(hist_path)):
        with open(hist_path, 'w', newline='') as fh:
            csv.writer(fh).writerow(['epoch', 'train_loss', 'train_ce', 'train_bow', 'val_ce',
                                     'val_bleu', 'lr_enc', 'lr_dec', 'minutes'])

    log.info(f"Train {len(train_ds)} clips, {len(train_dl)} steps/epoch | Val {len(val_ds)} clips")
    stop_reason = 'max epochs'

    for epoch in range(start_epoch, cfg['epochs'] + 1):
        frozen = epoch <= cfg['freeze_decoder_epochs']
        model.set_decoder_trainable(not frozen)
        if frozen:
            log.info(f"Epoch {epoch}: T5 decoder frozen, training encoder only")
        model.train()
        te, run, run_ce, run_bow = time.time(), [], [], []
        for step, b in enumerate(train_dl, 1):
            loss = model(b['features'].to(device, non_blocking=True),
                         b['feature_mask'].to(device, non_blocking=True),
                         b['labels'].to(device, non_blocking=True),
                         cfg['label_smoothing'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            opt.step(); sched.step()
            run.append(loss.item())
            run_ce.append(model.parts['ce']); run_bow.append(model.parts.get('bow', 0.0))
            if step % 200 == 0:
                log.info(f"  epoch {epoch} step {step}/{len(train_dl)} loss {np.mean(run[-200:]):.3f} "
                         f"(ce {np.mean(run_ce[-200:]):.3f}, bow {np.mean(run_bow[-200:]):.3f})")

        ev = evaluate(model, val_dl, tok, device, cfg['num_beams_eval'],
                      cfg['max_text_len'], cfg['label_smoothing'])
        lrs = [g['lr'] for g in opt.param_groups]
        mins = (time.time() - te) / 60
        log.info(f"Epoch {epoch}: train {np.mean(run):.3f} (ce {np.mean(run_ce):.3f}, "
                 f"bow {np.mean(run_bow):.3f}) | val ce {ev['loss']:.3f} | "
                 f"val BLEU {ev['bleu']:.2f} | {mins:.1f} min")
        for h, r in list(zip(ev['hyps'], ev['refs']))[:3]:
            log.info(f"    REF: {r}\n                  HYP: {h}")
        with open(hist_path, 'a', newline='') as fh:
            csv.writer(fh).writerow([epoch, round(np.mean(run), 4), round(np.mean(run_ce), 4),
                                     round(np.mean(run_bow), 4), round(ev['loss'], 4),
                                     round(ev['bleu'], 3), lrs[0], lrs[1], round(mins, 2)])

        if ev['bleu'] > best_bleu:
            best_bleu, bad_epochs = ev['bleu'], 0
            save_ckpt(os.path.join(cfg['out_dir'], 'best.pt'), model, opt, sched,
                      epoch, best_bleu, bad_epochs, cfg)
            log.info(f"  ★ new best BLEU {best_bleu:.2f}, saved best.pt")
        else:
            bad_epochs += 1
        save_ckpt(os.path.join(cfg['out_dir'], 'last.pt'), model, opt, sched,
                  epoch, best_bleu, bad_epochs, cfg)

        if bad_epochs >= cfg['patience']:
            stop_reason = f'no BLEU gain for {bad_epochs} epochs'; break
        hours = (time.time() - t0) / 3600
        if hours + mins * 1.5 / 60 > cfg['max_hours']:
            stop_reason = f'time limit ({hours:.1f} h used)'; break

    log.info(f"Training stopped: {stop_reason}. Best val BLEU {best_bleu:.2f}")
    return {'model': model, 'tok': tok, 'best_bleu': best_bleu, 'stop_reason': stop_reason}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — FINAL TEST
# ─────────────────────────────────────────────────────────────────────────────

def test(cfg: dict, model, tok, test_sets: Optional[Dict[str, How2SignS2T]] = None) -> Dict:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best = os.path.join(cfg['out_dir'], 'best.pt')
    ck = torch.load(best, map_location=device, weights_only=False)
    model.load_state_dict(ck['model']); model.to(device)
    log.info(f"Testing best.pt (epoch {ck['epoch']}, val BLEU {ck['best_bleu']:.2f})")

    if test_sets is None:
        test_sets = {
            'test_full':    make_dataset(cfg, 'test', augment=False, clean=False, preload=False),
            'test_cleaned': make_dataset(cfg, 'test', augment=False, clean=True, preload=False),
        }
    results = {}
    for name, ds in test_sets.items():
        ev = evaluate(model, make_loader(ds, tok, cfg, shuffle=False), tok, device,
                      cfg['num_beams_test'], cfg['max_text_len'])
        results[name] = ev['bleu']
        log.info(f"{name}: {len(ds)} clips | BLEU-4 {ev['bleu']:.2f} (beam {cfg['num_beams_test']})")
        if name == 'test_full':
            with open(os.path.join(cfg['out_dir'], 'test_predictions.csv'), 'w', newline='') as fh:
                w = csv.writer(fh); w.writerow(['sentence_name', 'reference', 'prediction'])
                w.writerows(zip(ev['sids'], ev['refs'], ev['hyps']))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

class _FakeTok:
    """Word-level toy tokenizer so the smoke test runs offline."""
    def __init__(self):
        self.vocab = {'<pad>': 0, '</s>': 1}
    def _id(self, w):
        return self.vocab.setdefault(w, len(self.vocab)) % 100
    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        ids = [[self._id(w) for w in t.lower().split()][: max_length - 1] + [1] for t in texts]
        L = max(map(len, ids))
        ii = torch.tensor([r + [0] * (L - len(r)) for r in ids])
        return {'input_ids': ii, 'attention_mask': (ii != 0).long()}
    def batch_decode(self, ids, skip_special_tokens=True):
        inv = {v % 100: k for k, v in self.vocab.items()}
        return [' '.join(inv.get(int(i), '?') for i in r if int(i) > 1) for r in ids]


def smoke() -> None:
    import tempfile
    import pandas as pd
    from transformers import T5Config, T5ForConditionalGeneration
    from s2t_phase1_dataset import _fake_clip, FILE_SUFFIX

    print("\n" + "=" * 65 + "\n  Sign2Text Phase 5 — Smoke Test\n" + "=" * 65 + "\n")
    rng = np.random.default_rng(0)
    words = 'hello my name is sign today we learn how to cook fish'.split()

    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, 'metadata'))
        for split, n in (('train', 40), ('val', 8), ('test', 8)):
            d = os.path.join(root, split, 'frontal'); os.makedirs(d)
            rows = []
            for i in range(n):
                name = f'{split}{i:03d}-1-rgb_front'
                np.save(os.path.join(d, name + FILE_SUFFIX), _fake_clip(rng, int(rng.integers(40, 120))))
                rows.append({'SENTENCE_NAME': name,
                             'SENTENCE': ' '.join(rng.choice(words, int(rng.integers(3, 8))))})
            pd.DataFrame(rows).to_csv(os.path.join(root, 'metadata', f'how2sign_realigned_{split}.csv'),
                                      sep='\t', index=False)

        out = os.path.join(root, 'out')
        cfg = dict(CONFIG, data_root=root, out_dir=out, batch_size=8, epochs=2,
                   warmup_steps=5, num_workers=0, num_beams_test=2, freeze_decoder_epochs=1)

        def tiny():
            c = T5Config(vocab_size=100, d_model=64, d_ff=128, d_kv=16, num_heads=4,
                         num_layers=2, num_decoder_layers=2,
                         pad_token_id=0, eos_token_id=1, decoder_start_token_id=0)
            return Sign2Text(T5ForConditionalGeneration(c), aux_weight=0.5,
                             d=32, layers=2, heads=4, ff=64)

        tok = _FakeTok()
        print("── Test 1: Train 2 epochs")
        r = train(cfg, model=tiny(), tok=tok)
        for f in ('best.pt', 'last.pt', 'history.csv', 'train_pre.npz', 'val_pre.npz'):
            assert os.path.exists(os.path.join(out, f)), f
        assert len(pd.read_csv(os.path.join(out, 'history.csv'))) == 2
        print("  checkpoints, history, preload caches saved ✓")

        print("── Test 2: Resume continues from epoch 3")
        cfg3 = dict(cfg, epochs=3)
        train(cfg3, model=tiny(), tok=tok)
        h = pd.read_csv(os.path.join(out, 'history.csv'))
        assert h['epoch'].tolist() == [1, 2, 3], h['epoch'].tolist()
        print("  resumed, history has epochs 1-3 ✓")

        print("── Test 3: Final test (full + cleaned)")
        res = test(cfg3, tiny(), tok)
        assert set(res) == {'test_full', 'test_cleaned'}
        assert os.path.exists(os.path.join(out, 'test_predictions.csv'))
        print(f"  {res} ✓")

    print("\n" + "=" * 65 + "\n  All tests PASSED ✓\n" + "=" * 65)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        smoke()
    else:
        r = train(CONFIG)
        test(CONFIG, r['model'], r['tok'])
