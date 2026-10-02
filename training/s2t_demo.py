"""
Sign2Text Demo — Phrase classifier
==================================
Trains a classifier for the recorded phrases (demo/record_phrases.py),
starting from the pose encoder trained on How2Sign (run 2).

Pipeline per take:
    raw keypoints (T, 543, 3) at ~16 fps
    → resample to 20 fps (How2Sign median)
    → trim: keep only the part where a hand is raised (signing)
    → same features as How2Sign: stride 2, 85 points, 384 per frame
    → pose encoder (from run 2) → mean + max pooling → class

The same trimming rule is used live (Segmenter), so training and the
webcam demo see the same kind of clip.

Train (laptop CPU is fine, from the repo root):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    python training/s2t_demo.py --data data --pretrained results/run2_weights.pt

Compare with no pretraining:
    python training/s2t_demo.py --data data --no-pretrain --out results/demo_scratch.pt

Tests:
    python training/s2t_demo.py --selftest
"""

import os
import csv
import sys
import time
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s2t_phase1_dataset import select_stride, features_from_selected, ASPECT, FEATURE_DIM
from s2t_phase34_model import PoseEncoder

TARGET_FPS  = 20.0                 # How2Sign median fps
STRIDE      = 2                    # same as How2Sign training
L_SH, R_SH  = 11, 12               # pose shoulders
LH_WRIST, RH_WRIST = 501, 522      # hand wrists (How2Sign layout)
RAISE_LIMIT = 1.3                  # hand counts as "up" if wrist is less than this many
                                   # shoulder widths below the shoulder line
PAD_FRAMES  = 3                    # frames kept before / after the active part
MIN_SEG     = 6                    # shortest segment (frames at 20 fps)
END_GAP     = 8                    # live: inactive frames that end a segment (0.4 s)

DEFAULT_ENCODER = dict(d=384, layers=6, heads=6, ff=1536, dropout=0.15)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — SHARED HELPERS (training + live demo)
# ─────────────────────────────────────────────────────────────────────────────

def resample_fps(kp: np.ndarray, t: np.ndarray, fps: float = TARGET_FPS) -> np.ndarray:
    """Nearest-frame resample by timestamp to a fixed fps."""
    if len(kp) < 2:
        return kp
    grid = np.arange(t[0], t[-1] + 1e-6, 1.0 / fps)
    idx = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    prev = np.clip(idx - 1, 0, len(t) - 1)
    idx = np.where(np.abs(t[prev] - grid) < np.abs(t[idx] - grid), prev, idx)
    return kp[idx]


def active_frames(kp: np.ndarray) -> np.ndarray:
    """
    (T, 543, 3) → (T,) bool: a hand is detected AND raised (wrist not far
    below the shoulders). Resting hands are usually low or out of frame.
    """
    ls, rs = kp[:, L_SH], kp[:, R_SH]
    sh_ok = (np.abs(ls).sum(1) > 0) & (np.abs(rs).sum(1) > 0)
    if sh_ok.any():
        sh_y  = np.median(((ls[:, 1] + rs[:, 1]) / 2)[sh_ok])
        dx    = (ls[sh_ok, 0] - rs[sh_ok, 0]) * ASPECT
        width = max(float(np.median(np.hypot(dx, ls[sh_ok, 1] - rs[sh_ok, 1]))), 1e-3)
    else:
        sh_y, width = 0.5, 0.25
    limit = sh_y + RAISE_LIMIT * width
    act = np.zeros(len(kp), dtype=bool)
    for w in (LH_WRIST, RH_WRIST):
        present = np.abs(kp[:, w:w + 21]).sum(axis=(1, 2)) > 0
        act |= present & (kp[:, w, 1] < limit)
    return act


def trim_active(kp: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Keep first..last active frame (+ padding). Falls back to the whole clip."""
    act = np.where(active_frames(kp))[0]
    if len(act) == 0 or act[-1] - act[0] + 1 < MIN_SEG:
        if rng is not None and len(kp) > 40:            # idle-like: random 1-2 s window
            L = int(rng.integers(20, 41))
            s = int(rng.integers(0, len(kp) - L + 1))
            return kp[s:s + L]
        return kp
    pad_a = PAD_FRAMES + (int(rng.integers(-2, 3)) if rng is not None else 0)
    pad_b = PAD_FRAMES + (int(rng.integers(-2, 3)) if rng is not None else 0)
    a = max(0, act[0] - max(0, pad_a))
    b = min(len(kp), act[-1] + 1 + max(0, pad_b))
    return kp[a:b]


def clip_features(kp: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Trimmed 20 fps clip (T, 543, 3) → features (T', 384)."""
    x, p = select_stride(kp.astype(np.float32), STRIDE)
    f = features_from_selected(x, p, max_frames=128, rng=rng)
    if f is None:                                        # no shoulders at all
        f = np.zeros((max(1, len(x)), FEATURE_DIM), np.float32)
    return f


class Segmenter:
    """
    Live: feed one (543, 3) frame at a time (already at ~20 fps or not;
    timestamps are used). Returns a finished segment (T, 543, 3) when the
    signer lowers their hands, else None.
    """

    def __init__(self, ref_frames: int = 40):
        self.history: List[np.ndarray] = []     # recent frames, for shoulder reference
        self.seg: List[np.ndarray] = []
        self.seg_t: List[float] = []
        self.pre: List[Tuple[np.ndarray, float]] = []
        self.inactive = 0
        self.ref_frames = ref_frames

    def push(self, kp: np.ndarray, t: float) -> Optional[np.ndarray]:
        self.history = (self.history + [kp])[-self.ref_frames:]
        ref = np.stack(self.history)
        act = bool(active_frames(ref)[-1])

        if not self.seg:
            self.pre = (self.pre + [(kp, t)])[-PAD_FRAMES:]
            if act:
                self.seg = [f for f, _ in self.pre]
                self.seg_t = [s for _, s in self.pre]
                self.inactive = 0
            return None

        self.seg.append(kp); self.seg_t.append(t)
        self.inactive = 0 if act else self.inactive + 1
        if self.inactive >= END_GAP or self.seg_t[-1] - self.seg_t[0] > 8.0:
            seg = resample_fps(np.stack(self.seg), np.array(self.seg_t))
            self.seg, self.seg_t, self.pre, self.inactive = [], [], [], 0
            seg = trim_active(seg)
            return seg if len(seg) >= MIN_SEG else None
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — MODEL
# ─────────────────────────────────────────────────────────────────────────────

class PhraseClassifier(nn.Module):
    def __init__(self, n_classes: int, encoder_cfg: dict = None, d_out: int = 512):
        super().__init__()
        self.encoder_cfg = dict(encoder_cfg or DEFAULT_ENCODER)
        self.encoder = PoseEncoder(out_dim=d_out, **self.encoder_cfg)
        self.head = nn.Sequential(nn.LayerNorm(2 * d_out), nn.Dropout(0.3),
                                  nn.Linear(2 * d_out, n_classes))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, m = self.encoder(x, mask)
        mf = m.unsqueeze(-1).float()
        mean = (h * mf).sum(1) / mf.sum(1).clamp(min=1)
        mx = h.masked_fill(~m.unsqueeze(-1), -1e4).max(1).values
        return self.head(torch.cat([mean, mx], -1))


def load_pretrained_encoder(model: PhraseClassifier, path: str) -> int:
    """Copy 'encoder.*' weights from a run 2 checkpoint. Returns tensors loaded."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = {k[len('encoder.'):]: v.float() for k, v in ck['model'].items() if k.startswith('encoder.')}
    missing, unexpected = model.encoder.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"Encoder shape mismatch, missing: {missing[:5]}")
    return len(sd)


def encoder_cfg_from(path: Optional[str]) -> dict:
    if path and os.path.exists(path):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        cfg = ck.get('cfg', {}).get('encoder')
        if cfg:
            return dict(cfg)
    return dict(DEFAULT_ENCODER)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_takes(data_dir: str) -> Tuple[List[dict], List[str]]:
    """Read index.csv + npz files. Returns takes and class names (by class_id)."""
    rows, names = [], {}
    with open(os.path.join(data_dir, 'index.csv')) as fh:
        for r in csv.DictReader(fh):
            path = os.path.join(data_dir, r['file'].replace('\\', os.sep).replace('/', os.sep))
            if not os.path.exists(path):
                continue
            with np.load(path) as z:
                kp = resample_fps(z['kp'], z['t'])
            cid = int(r['class_id'])
            names[cid] = r['phrase']
            rows.append({'kp': kp, 'y': cid, 'session': r['session'], 'file': r['file']})
    classes = [names.get(i, f'class_{i}') for i in range(max(names) + 1)]
    return rows, classes


def split_takes(takes: List[dict], val_frac: float = 0.2, seed: int = 0):
    """
    Val = the latest session if there are 2+ sessions and it covers every
    class (honest: a different day). Otherwise a random 20% per class.
    """
    sessions = sorted({t['session'] for t in takes})
    if len(sessions) >= 2:
        last = sessions[-1]
        val = [t for t in takes if t['session'] == last]
        if {t['y'] for t in val} == {t['y'] for t in takes}:
            return [t for t in takes if t['session'] != last], val, f'session {last}'
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in sorted({t['y'] for t in takes}):
        items = [t for t in takes if t['y'] == c]
        rng.shuffle(items)
        k = max(1, int(round(len(items) * val_frac)))
        va += items[:k]; tr += items[k:]
    return tr, va, f'random {int(val_frac * 100)}% per class'


def make_batch(takes: List[dict], rng: Optional[np.random.Generator]):
    feats = [torch.from_numpy(clip_features(trim_active(t['kp'], rng), rng)) for t in takes]
    T = max(f.shape[0] for f in feats)
    x = torch.zeros(len(feats), T, FEATURE_DIM)
    m = torch.zeros(len(feats), T, dtype=torch.bool)
    for i, f in enumerate(feats):
        x[i, :len(f)] = f; m[i, :len(f)] = True
    y = torch.tensor([t['y'] for t in takes])
    return x, m, y


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — TRAINING
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, takes, n_classes, batch=32):
    model.eval()
    preds = []
    for i in range(0, len(takes), batch):
        x, m, _ = make_batch(takes[i:i + batch], None)
        preds += model(x, m).argmax(-1).tolist()
    y = [t['y'] for t in takes]
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for a, b in zip(y, preds):
        cm[a, b] += 1
    return float(np.mean(np.array(preds) == np.array(y))), cm


def train(data_dir: str, out: str, pretrained: Optional[str], epochs: int = 80,
          freeze_epochs: int = 10, batch: int = 16, patience: int = 20, seed: int = 0,
          verbose: bool = True) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    log = print if verbose else (lambda *a, **k: None)

    takes, classes = load_takes(data_dir)
    tr, va, how = split_takes(takes, seed=seed)
    n = len(classes)
    log(f"{len(takes)} takes, {n} classes | train {len(tr)} | val {len(va)} ({how})")

    model = PhraseClassifier(n, encoder_cfg_from(pretrained))
    if pretrained:
        k = load_pretrained_encoder(model, pretrained)
        log(f"Loaded {k} encoder tensors from {pretrained}")
    else:
        freeze_epochs = 0
        log("No pretraining: encoder starts from random weights")

    head_opt = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=0.01)
    full_opt = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 1e-4},
        {'params': model.head.parameters(), 'lr': 5e-4},
    ], weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, bad, t0 = -1.0, 0, time.time()
    for ep in range(1, epochs + 1):
        frozen = ep <= freeze_epochs
        for p in model.encoder.parameters():
            p.requires_grad = not frozen
        opt = head_opt if frozen else full_opt
        model.train()
        order = rng.permutation(len(tr))
        losses = []
        for i in range(0, len(order), batch):
            x, m, y = make_batch([tr[j] for j in order[i:i + batch]], rng)
            loss = loss_fn(model(x, m), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        acc, cm = evaluate(model, va, n)
        tag = ''
        if acc > best:
            best, bad, tag = acc, 0, '  ★ saved'
            torch.save({'model': model.state_dict(), 'classes': classes,
                        'encoder_cfg': model.encoder_cfg, 'target_fps': TARGET_FPS,
                        'stride': STRIDE, 'val_acc': acc, 'epoch': ep,
                        'pretrained': bool(pretrained)}, out)
        else:
            bad += 1
        log(f"epoch {ep:3d} {'(encoder frozen) ' if frozen else ''}loss {np.mean(losses):.3f} | "
            f"val acc {100 * acc:5.1f}%{tag}")
        if bad >= patience and not frozen:
            log(f"Stopped: no gain for {patience} epochs"); break

    ck = torch.load(out, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    acc, cm = evaluate(model, va, n)
    log(f"\nBest val accuracy: {100 * acc:.1f}% (epoch {ck['epoch']}) | {(time.time() - t0) / 60:.1f} min")
    log("\nPer phrase (val):")
    for i, c in enumerate(classes):
        tot = cm[i].sum()
        if tot == 0:
            continue
        wrong = [(classes[j], cm[i, j]) for j in np.argsort(-cm[i]) if j != i and cm[i, j] > 0][:2]
        miss = ', '.join(f"{w} ×{k}" for w, k in wrong)
        log(f"  {c:22s} {cm[i, i]}/{tot}" + (f"   confused with: {miss}" if miss else ''))
    log(f"\nSaved {out}")
    return {'val_acc': acc, 'confusion': cm, 'classes': classes}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — SELF-TEST (fake data)
# ─────────────────────────────────────────────────────────────────────────────

def _fake_take(rng, cls: int, T: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Shoulders fixed, hands down (missing) then raised, moving in a class pattern."""
    kp = np.zeros((T, 543, 3), np.float32)
    kp[:, 0:33] = rng.uniform(0.4, 0.6, (T, 33, 3))
    kp[:, L_SH] = [0.40, 0.55, 0]; kp[:, R_SH] = [0.60, 0.55, 0]
    kp[:, 33:501] = rng.uniform(0.45, 0.55, (T, 468, 3))
    a, b = T // 4, 3 * T // 4
    tt = np.linspace(0, 1, b - a)
    for w, side in ((LH_WRIST, -1), (RH_WRIST, 1)):
        base = np.array([0.5 + side * 0.08, 0.6, 0])
        hand = base + rng.normal(0, 0.01, (21, 3))
        move = np.stack([np.sin(2 * np.pi * (cls + 1) * tt) * 0.05,
                         np.cos(np.pi * (cls + 1) * tt) * 0.05, np.zeros_like(tt)], 1)
        kp[a:b, w:w + 21] = hand[None] + move[:, None]
        kp[a:b, w:w + 21, 2] = 0
    t = np.arange(T) / 16.3
    return kp, t


def selftest() -> None:
    import tempfile
    print("\n" + "=" * 60 + "\n  Sign2Text demo classifier — self-test\n" + "=" * 60)
    rng = np.random.default_rng(0)

    print("── Test 1: resample 16.3 → 20 fps")
    kp, t = _fake_take(rng, 0, 98)
    r = resample_fps(kp, t)
    assert abs(len(r) - (t[-1] * 20 + 1)) <= 1, len(r)
    print(f"  {len(kp)} frames → {len(r)} ✓")

    print("── Test 2: trimming keeps the raised-hand part")
    act = active_frames(r)
    tr_ = trim_active(r)
    assert act[len(r) // 2] and not act[0] and len(tr_) < len(r) and len(tr_) >= act.sum()
    print(f"  {len(r)} → {len(tr_)} frames ✓")

    print("── Test 3: live segmenter finds one segment per sign")
    seg = Segmenter()
    found = []
    for k in range(2):
        kp, t = _fake_take(rng, 1, 100)
        for f, s in zip(kp, t + k * 10):
            out = seg.push(f, s)
            if out is not None:
                found.append(len(out))
    assert len(found) == 2, found
    print(f"  segments {found} ✓")

    print("── Test 4: train on fake data (3 classes, 2 sessions), save, reload")
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, 'x'))
        with open(os.path.join(d, 'index.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['file', 'class_id', 'phrase', 'frames', 'seconds', 'fps', 'hand_pct', 'session'])
            for c in range(3):
                for i in range(8):
                    kp, t = _fake_take(rng, c)
                    f = f'x\\c{c}_{i}.npz'
                    np.savez(os.path.join(d, 'x', f'c{c}_{i}.npz'), kp=kp, t=t)
                    w.writerow([f, c, f'phrase{c}', len(kp), 6, 16.3, 80, 's1' if i < 6 else 's2'])
        # fake "run 2" checkpoint with a small encoder
        small = dict(d=32, layers=2, heads=4, ff=64, dropout=0.1)
        enc = PoseEncoder(out_dim=512, **small)
        torch.save({'model': {f'encoder.{k}': v.half() for k, v in enc.state_dict().items()},
                    'cfg': {'encoder': small}}, os.path.join(d, 'pre.pt'))
        out = os.path.join(d, 'demo.pt')
        res = train(d, out, os.path.join(d, 'pre.pt'), epochs=25, freeze_epochs=3,
                    patience=25, verbose=False)
        ck = torch.load(out, map_location='cpu', weights_only=False)
        m2 = PhraseClassifier(len(ck['classes']), ck['encoder_cfg'])
        m2.load_state_dict(ck['model'])
        assert ck['classes'] == ['phrase0', 'phrase1', 'phrase2']
        assert res['val_acc'] >= 0.5, res['val_acc']
        print(f"  val acc {100 * res['val_acc']:.0f}% on fake data, checkpoint reloads ✓")

    print("\nAll tests PASSED ✓")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--pretrained', default='results/run2_weights.pt')
    ap.add_argument('--no-pretrain', action='store_true')
    ap.add_argument('--out', default='results/demo_model.pt')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        pre = None if a.no_pretrain else a.pretrained
        if pre and not os.path.exists(pre):
            sys.exit(f"Pretrained weights not found: {pre}\nUse --pretrained <path> or --no-pretrain")
        train(a.data, a.out, pre, epochs=a.epochs)
