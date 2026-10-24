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

Two signers (each has their own data folder):
    # unseen-signer test: train on Inaaya, test on the teammate
    python training/s2t_demo.py --data data --test data_friend
    # test an already trained model on a folder, no training
    python training/s2t_demo.py --eval-only results/demo_model.pt --test data_friend
    # train on both signers
    python training/s2t_demo.py --data data data_friend

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
from s2t_phase1_dataset import (select_stride, features_from_selected, ASPECT, FEATURE_DIM,
                                 POSE_KEEP, LIPS_OUTER, BROWS, LH_SL, RH_SL, FACE_SL)
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


def _mirror_perm() -> np.ndarray:
    """
    Index map that mirrors the 85 selected points left ↔ right:
    pose pairs swap, left hand ↔ right hand, lips and eyebrows swap sides.
    """
    pose_pairs = {11: 12, 13: 14, 15: 16, 17: 18, 19: 20, 21: 22}
    pm = {**pose_pairs, **{v: k for k, v in pose_pairs.items()}}
    pose = [POSE_KEEP.index(pm.get(i, i)) for i in POSE_KEEP]
    lh = list(range(RH_SL.start, RH_SL.stop))          # new left hand  = old right hand
    rh = list(range(LH_SL.start, LH_SL.stop))          # new right hand = old left hand
    face_pairs = {61: 291, 185: 409, 40: 270, 39: 269, 37: 267, 0: 0, 146: 375, 91: 321,
                  181: 405, 84: 314, 17: 17, 70: 300, 63: 293, 105: 334, 66: 296, 107: 336}
    fm = {**face_pairs, **{v: k for k, v in face_pairs.items()}}
    face_list = LIPS_OUTER + BROWS
    face = [FACE_SL.start + face_list.index(fm[i]) for i in face_list]
    return np.array(pose + lh + rh + face)


MIRROR_PERM = _mirror_perm()


def mirror_selected(x: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mirror a clip left ↔ right, as if a left-handed signer signed with the right
    hand (or the other way). ASL lets each signer pick their dominant hand.
    """
    x = x[:, MIRROR_PERM].copy()
    x[..., 0] *= -1.0
    return x, p[:, MIRROR_PERM].copy()


def clip_features(kp: np.ndarray, rng: Optional[np.random.Generator] = None,
                  mirror: bool = False) -> np.ndarray:
    """Trimmed 20 fps clip (T, 543, 3) → features (T', 384)."""
    x, p = select_stride(kp.astype(np.float32), STRIDE)
    if mirror:
        x, p = mirror_selected(x, p)
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

def load_takes(data_dirs, verbose: bool = False) -> Tuple[List[dict], List[str]]:
    """
    Read index.csv + npz files from one or more data folders (one folder per
    signer). Returns takes and class names (by class_id).
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    rows, names = [], {}
    for data_dir in data_dirs:
        n0 = len(rows)
        rows += _load_dir(data_dir, names)
        if verbose:
            print(f"  {data_dir}: {len(rows) - n0} takes")
    if not rows:
        raise SystemExit(f"No takes found in {data_dirs}")
    classes = [names.get(i, f'class_{i}') for i in range(max(names) + 1)]
    return rows, classes


def _load_dir(data_dir: str, names: dict) -> List[dict]:
    rows = []
    index = os.path.join(data_dir, 'index.csv')
    if not os.path.exists(index):
        raise SystemExit(f"Missing {index}. Is '{data_dir}' a recording folder?")
    signer = os.path.basename(os.path.normpath(data_dir))
    with open(index) as fh:
        for r in csv.DictReader(fh):
            path = os.path.join(data_dir, r['file'].replace('\\', os.sep).replace('/', os.sep))
            if not os.path.exists(path):
                continue
            with np.load(path) as z:
                kp = resample_fps(z['kp'], z['t'])
            cid = int(r['class_id'])
            if cid in names and names[cid] != r['phrase']:
                raise SystemExit(f"Class {cid} is '{names[cid]}' in one folder and "
                                 f"'{r['phrase']}' in {data_dir}. Both signers must use "
                                 f"the same PHRASES list.")
            names[cid] = r['phrase']
            rows.append({'kp': kp, 'y': cid, 'phrase': r['phrase'],
                         'session': f"{signer}/{r['session']}",
                         'signer': signer, 'file': r['file']})
    return rows


def split_takes(takes: List[dict], val_frac: float = 0.2, seed: int = 0):
    """
    Val = the latest session (a different day is the honest test), but only if
    it covers every class, holds at most 40% of the takes, and leaves at least
    2 training takes per class. Otherwise a random 20% per class.
    """
    from collections import Counter
    sessions = sorted({t['session'] for t in takes}, key=lambda x: x.split('/')[-1])   # by time
    all_classes = {t['y'] for t in takes}
    if len(sessions) >= 2:
        last = sessions[-1]
        val = [t for t in takes if t['session'] == last]
        tr = [t for t in takes if t['session'] != last]
        per_class = Counter(t['y'] for t in tr)
        if ({t['y'] for t in val} == all_classes and len(val) <= 0.4 * len(takes)
                and all(per_class[c] >= 2 for c in all_classes)):
            return tr, val, f'session {last}'
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in sorted({t['y'] for t in takes}):
        items = [t for t in takes if t['y'] == c]
        rng.shuffle(items)
        k = max(1, int(round(len(items) * val_frac)))
        va += items[:k]; tr += items[k:]
    return tr, va, f'random {int(val_frac * 100)}% per class'


def make_batch(takes: List[dict], rng: Optional[np.random.Generator], mirror: Optional[bool] = None):
    """
    mirror=None: random mirror for half the takes when training (rng given), none otherwise.
    mirror=True/False: mirror all / none (used for two-way prediction).
    """
    feats = []
    for t in takes:
        m = (rng.random() < 0.5) if (mirror is None and rng is not None) else bool(mirror)
        feats.append(torch.from_numpy(clip_features(trim_active(t['kp'], rng), rng, mirror=m)))
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

def _pad(feats: List[np.ndarray]):
    T = max(len(f) for f in feats)
    x = torch.zeros(len(feats), T, FEATURE_DIM)
    m = torch.zeros(len(feats), T, dtype=torch.bool)
    for i, f in enumerate(feats):
        x[i, :len(f)] = torch.from_numpy(f); m[i, :len(f)] = True
    return x, m


@torch.no_grad()
def predict_proba(model, segments: List[np.ndarray]) -> torch.Tensor:
    """
    Class probabilities for trimmed clips (T, 543, 3). Averages the clip and its
    mirror image, so left- and right-handed signing are treated the same.
    """
    model.eval()
    probs = 0
    for mirror in (False, True):
        x, m = _pad([clip_features(s, None, mirror=mirror) for s in segments])
        probs = probs + torch.softmax(model(x, m), -1)
    return probs / 2


@torch.no_grad()
def evaluate(model, takes, n_classes, batch=32):
    model.eval()
    preds = []
    for i in range(0, len(takes), batch):
        segs = [trim_active(t['kp']) for t in takes[i:i + batch]]
        preds += predict_proba(model, segs).argmax(-1).tolist()
    y = [t['y'] for t in takes]
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for a, b in zip(y, preds):
        cm[a, b] += 1
    return float(np.mean(np.array(preds) == np.array(y))), cm


def report(model, takes, classes, title: str, log=print) -> float:
    acc, cm = evaluate(model, takes, len(classes))
    log(f"\n{title}: {100 * acc:.1f}% of {len(takes)} takes")
    for i, c in enumerate(classes):
        tot = cm[i].sum()
        if tot == 0:
            continue
        wrong = [(classes[j], cm[i, j]) for j in np.argsort(-cm[i]) if j != i and cm[i, j] > 0][:2]
        miss = ', '.join(f"{w} ×{k}" for w, k in wrong)
        log(f"  {c:22s} {cm[i, i]}/{tot}" + (f"   confused with: {miss}" if miss else ''))
    return acc


def load_model(path: str):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = PhraseClassifier(len(ck['classes']), ck['encoder_cfg'])
    m.load_state_dict(ck['model'])
    m.eval()
    return m, ck


def check_phrases(takes: List[dict], classes: List[str], where: str) -> None:
    """Every test take's phrase must match the model's phrase for that class id."""
    bad = sorted({(t['y'], t['phrase']) for t in takes
                  if t['y'] >= len(classes) or classes[t['y']] != t['phrase']})
    if bad:
        cid, ph = bad[0]
        expected = classes[cid] if cid < len(classes) else 'nothing'
        raise SystemExit(f"Phrase list mismatch in {where}: class {cid} is '{ph}' there "
                         f"but '{expected}' in the model. Use the same PHRASES list.")


def eval_only(model_path: str, test_dirs) -> float:
    model, ck = load_model(model_path)
    takes, _ = load_takes(test_dirs, verbose=True)
    check_phrases(takes, ck['classes'], ', '.join(test_dirs))
    return report(model, takes, ck['classes'], f"Test accuracy on {', '.join(test_dirs)}")


def train(data_dir, out: str, pretrained: Optional[str], epochs: int = 80,
          freeze_epochs: int = 10, batch: int = 16, patience: int = 20, seed: int = 0,
          verbose: bool = True, test_dirs=None) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    log = print if verbose else (lambda *a, **k: None)

    log("Training data:")
    takes, classes = load_takes(data_dir, verbose=verbose)
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
    report(model, va, classes, "Validation (per phrase)", log)
    test_acc = None
    if test_dirs:
        log("\nUnseen-signer test data:")
        test_takes, _ = load_takes(test_dirs, verbose=verbose)
        check_phrases(test_takes, classes, ', '.join(test_dirs))
        test_acc = report(model, test_takes, classes, "Unseen-signer test", log)
    log(f"\nSaved {out}")
    return {'val_acc': acc, 'test_acc': test_acc, 'confusion': cm, 'classes': classes}


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

    print("── Test 5: two signers, unseen-signer test, eval-only, phrase-list check")
    with tempfile.TemporaryDirectory() as d:
        def make(folder, n, names=('phrase0', 'phrase1', 'phrase2')):
            os.makedirs(os.path.join(d, folder, 'x'))
            with open(os.path.join(d, folder, 'index.csv'), 'w', newline='') as fh:
                w = csv.writer(fh)
                w.writerow(['file', 'class_id', 'phrase', 'frames', 'seconds', 'fps', 'hand_pct', 'session'])
                for c in range(3):
                    for i in range(n):
                        kp, t = _fake_take(rng, c)
                        np.savez(os.path.join(d, folder, 'x', f'c{c}_{i}.npz'), kp=kp, t=t)
                        w.writerow([f'x\\c{c}_{i}.npz', c, names[c], len(kp), 6, 16.3, 80, 's1'])
        make('data', 6); make('data_b', 3); make('data_bad', 2, ('phrase0', 'other', 'phrase2'))
        out = os.path.join(d, 'demo.pt')
        res = train(os.path.join(d, 'data'), out, None, epochs=8, patience=8, verbose=False,
                    test_dirs=[os.path.join(d, 'data_b')])
        assert res['test_acc'] is not None
        takes, _ = load_takes([os.path.join(d, 'data'), os.path.join(d, 'data_b')])
        assert len(takes) == 27 and {t['signer'] for t in takes} == {'data', 'data_b'}
        acc = eval_only(out, [os.path.join(d, 'data_b')])
        try:
            load_takes([os.path.join(d, 'data'), os.path.join(d, 'data_bad')])
            raise AssertionError('mismatch not caught')
        except SystemExit:
            pass
        print(f"  ✓ (test acc {100 * res['test_acc']:.0f}%, eval-only {100 * acc:.0f}%)")

    print("── Test 6: tiny first session does not swallow training; missing class is fine")
    fake = lambda y, s: {'y': y, 'session': s, 'phrase': f'p{y}'}
    takes = [fake(0, 'a/1')] + [fake(c, 'a/2') for c in range(3) for _ in range(3)]
    tr, va, how = split_takes(takes)
    assert how.startswith('random') and len(tr) >= 6, (how, len(tr))
    takes = [fake(c, 'a/1') for c in range(3) for _ in range(4)] + [fake(c, 'a/2') for c in range(3)]
    tr, va, how = split_takes(takes)
    assert how.startswith('session') and len(va) == 3
    check_phrases([fake(0, 'x'), fake(2, 'x')], ['p0', 'p1', 'p2'], 'test')   # class 1 missing: ok
    try:
        check_phrases([{'y': 1, 'phrase': 'other'}], ['p0', 'p1', 'p2'], 'test')
        raise AssertionError('mismatch not caught')
    except SystemExit:
        pass
    print("  ✓")

    print("── Test 7: mirroring")
    assert sorted(MIRROR_PERM.tolist()) == list(range(85))
    assert (MIRROR_PERM[MIRROR_PERM] == np.arange(85)).all()          # mirror twice = identity
    kp, t = _fake_take(rng, 2)
    kp[:, RH_WRIST:RH_WRIST + 21] = 0                                  # one-handed sign
    x, p = select_stride(resample_fps(kp, t), STRIDE)
    xm, pm = mirror_selected(x, p)
    assert pm[:, RH_SL].any() and not pm[:, LH_SL].any()               # hand moved to the other slot
    x2, p2 = mirror_selected(xm, pm)
    assert np.allclose(x2, x) and (p2 == p).all()
    f1 = features_from_selected(x, p, 128); f2 = features_from_selected(xm, pm, 128)
    assert f1.shape == f2.shape and not np.allclose(f1, f2)
    m3 = PhraseClassifier(3, dict(d=32, layers=2, heads=4, ff=64, dropout=0.1))
    pr = predict_proba(m3, [resample_fps(kp, t)])
    assert pr.shape == (1, 3) and abs(float(pr.sum()) - 1) < 1e-4
    print("  ✓ mirror map is a clean left/right swap, two-way prediction works")

    print("\nAll tests PASSED ✓")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', default=['data'], help='one or more recording folders')
    ap.add_argument('--test', nargs='+', default=None, help='folders used only for testing (unseen signer)')
    ap.add_argument('--eval-only', default=None, metavar='MODEL', help='test a trained model, no training')
    ap.add_argument('--pretrained', default='results/run2_weights.pt')
    ap.add_argument('--no-pretrain', action='store_true')
    ap.add_argument('--out', default='results/demo_model.pt')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.eval_only:
        if not a.test:
            sys.exit("--eval-only needs --test <folder>")
        eval_only(a.eval_only, a.test)
    else:
        pre = None if a.no_pretrain else a.pretrained
        if pre and not os.path.exists(pre):
            sys.exit(f"Pretrained weights not found: {pre}\nUse --pretrained <path> or --no-pretrain")
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        train(a.data, a.out, pre, epochs=a.epochs, test_dirs=a.test)