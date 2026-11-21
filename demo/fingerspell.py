"""
Sign2Text — Fingerspelling (letters A to Z)
===========================================
Recognises fingerspelled letters one by one, so names can be spelled:
hold each letter still for about half a second, lower your hands when done.

How it works:
    - only the signing hand is used (the raised hand; a left hand is mirrored,
      so left- and right-handed signers look the same)
    - hand points are taken relative to the wrist and scaled by hand size,
      so position and distance from the camera do not matter, but hand
      orientation does (H vs U, K vs P need it)
    - a short window of 8 frames adds the motion that J and Z need
    - a small classifier predicts the letter for every window
    - the live Speller accepts a letter when the prediction is steady

Commands (from the repo root):
    python demo/fingerspell.py record --data data_letters            # record letters
    python demo/fingerspell.py record --data data_letters_friend
    python demo/fingerspell.py record --data data_letters --only H I Y U M
    python demo/fingerspell.py stats  --data data_letters
    python demo/fingerspell.py train  --data data_letters data_letters_friend
    python demo/fingerspell.py train  --data data_letters --test data_letters_friend
    python demo/fingerspell.py live                                   # spell and see letters
    python demo/fingerspell.py selftest

Recording tips:
    Hold the letter steady in front of your shoulder while recording.
    J and Z move: repeat the movement slowly 2 or 3 times.
    Change the angle of your hand a little between takes.
"""

import os
os.environ.setdefault('GLOG_minloglevel', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import sys
import csv
import time
import random
import argparse
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'demo'))

LETTERS = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
MOTION_LETTERS = {'J', 'Z'}

ASPECT = 16 / 9
LH0, RH0, N_HAND = 501, 522, 21
L_SH, R_SH = 11, 12
TIPS = [4, 8, 12, 16, 20]
WINDOW = 8                              # frames per window (~0.5 s at 16 fps)
FEAT_DIM = 63 + 15 + 3                  # mean hand shape + fingertip motion + wrist motion

HOLD_S, MOTION_HOLD_S = 2.0, 3.0        # recording length per take
SKIP_S = 0.3                            # ignore the first frames of a take (hand settling)

STABLE = 5                              # live: same letter this many windows in a row
MIN_PROB = 0.6                          # live: minimum confidence to accept a letter
END_FRAMES = 10                         # live: frames without a raised hand that end the name


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def _present(kp: np.ndarray, start: int) -> bool:
    return bool(np.abs(kp[start:start + N_HAND]).sum() > 0)


def signing_hand(kp: np.ndarray) -> Optional[Tuple[np.ndarray, bool]]:
    """
    One frame (543, 3) → (21, 3) points of the signing hand and whether it is
    the left-hand slot, or None. If both hands are visible, the higher one wins.
    """
    l, r = _present(kp, LH0), _present(kp, RH0)
    if not l and not r:
        return None
    if l and r:
        use_left = bool(kp[LH0, 1] < kp[RH0, 1])
    else:
        use_left = l
    s = LH0 if use_left else RH0
    return kp[s:s + N_HAND].astype(np.float32), use_left


def shoulder_width(kp: np.ndarray) -> float:
    ls, rs = kp[L_SH], kp[R_SH]
    if np.abs(ls).sum() > 0 and np.abs(rs).sum() > 0:
        return max(float(np.hypot((ls[0] - rs[0]) * ASPECT, ls[1] - rs[1])), 1e-3)
    return 0.3


def hand_shape(hand: np.ndarray, mirror: bool) -> np.ndarray:
    """(21, 3) → (21, 3) wrist-relative, hand-size scaled, left hands mirrored."""
    h = hand.copy()
    h[:, 0] *= ASPECT
    if mirror:
        h[:, 0] *= -1.0
    h = h - h[0:1]
    size = max(float(np.linalg.norm(h[9, :2])), 1e-4)        # wrist → middle finger base
    return h / size


def window_features(frames: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    WINDOW frames (543, 3) → feature vector (FEAT_DIM,), or None if the
    signing hand is missing in more than half of them.
    """
    shapes, wrists, sw = [], [], []
    mirror = None
    for kp in frames:
        sh = signing_hand(kp)
        if sh is None:
            continue
        hand, is_left = sh
        mirror = is_left if mirror is None else mirror
        shapes.append(hand_shape(hand, mirror))
        w = hand[0].copy(); w[0] *= ASPECT * (-1 if mirror else 1)
        wrists.append(w); sw.append(shoulder_width(kp))
    if len(shapes) < max(2, len(frames) // 2):
        return None
    shapes = np.stack(shapes)
    mean = shapes.mean(0).reshape(-1)                                  # 63
    tip_motion = (shapes[-1, TIPS] - shapes[0, TIPS]).reshape(-1)      # 15
    wrist_motion = (wrists[-1] - wrists[0]) / float(np.median(sw))     # 3
    return np.concatenate([mean, tip_motion, wrist_motion]).astype(np.float32)


def take_windows(kp: np.ndarray, t: np.ndarray, stride: int = 2) -> List[np.ndarray]:
    """All windows of one recorded take (skipping the settling part)."""
    start = int(np.searchsorted(t, t[0] + SKIP_S)) if len(t) else 0
    out = []
    for a in range(start, len(kp) - WINDOW + 1, stride):
        f = window_features(list(kp[a:a + WINDOW]))
        if f is not None:
            out.append(f)
    return out


def augment(f: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small rotation in the image plane, scale and noise on the hand shape."""
    f = f.copy()
    ang = np.deg2rad(rng.uniform(-12, 12))
    c, s = np.cos(ang), np.sin(ang)
    for sl in (slice(0, 63), slice(63, 78), slice(78, 81)):
        v = f[sl].reshape(-1, 3)
        x, y = v[:, 0].copy(), v[:, 1].copy()
        v[:, 0], v[:, 1] = c * x - s * y, s * x + c * y
        f[sl] = v.reshape(-1)
    f[:63] *= rng.uniform(0.93, 1.07)
    f += rng.normal(0, 0.02, f.shape).astype(np.float32)
    return f


def finger_spread(f: np.ndarray) -> float:
    """Gap between index and middle fingertips (U: together, V: apart), hand-size units."""
    return float(np.linalg.norm(f[3 * 8:3 * 8 + 2] - f[3 * 12:3 * 12 + 2]))


def tip_move(f: np.ndarray, tip_point: int) -> float:
    """How far one fingertip moved inside the window (J: pinky, Z: index)."""
    k = 63 + 3 * TIPS.index(tip_point)
    return float(np.linalg.norm(f[k:k + 2]))


MOTION_TIP = {'J': 20, 'Z': 8}


def best_split(a: np.ndarray, b: np.ndarray) -> float:
    """Threshold that best separates values a (below) from b (above)."""
    cand = np.unique(np.concatenate([a, b]))
    accs = [((a < c).mean() + (b >= c).mean()) / 2 for c in cand]
    return float(cand[int(np.argmax(accs))])


def learn_rules(feats: np.ndarray, ys: np.ndarray) -> dict:
    """
    Two simple checks on top of the model, learned from the recordings:
      U vs V: decided by the index/middle fingertip gap.
      J, Z:   only accepted when that fingertip really moves (else I, D, ...).
    """
    rules = {}
    U, V = LETTERS.index('U'), LETTERS.index('V')
    su = np.array([finger_spread(f) for f in feats[ys == U]])
    sv = np.array([finger_spread(f) for f in feats[ys == V]])
    if len(su) and len(sv):
        rules['uv_split'] = best_split(su, sv)
        rules['uv_medians'] = (float(np.median(su)), float(np.median(sv)))
    static = np.array([LETTERS[y] not in MOTION_LETTERS for y in ys])
    for L, tip in MOTION_TIP.items():
        if static.any():
            still = np.array([tip_move(f, tip) for f in feats[static]])
            rules[f'move_{L}'] = float(np.percentile(still, 90))      # normal shake of a held letter
    return rules


def apply_rules(rules: dict, f: np.ndarray, p: np.ndarray) -> Tuple[str, float]:
    p = p.copy()
    if rules:
        for L, tip in MOTION_TIP.items():
            k = f'move_{L}'
            if k in rules and tip_move(f, tip) < rules[k]:
                p[LETTERS.index(L)] = 0.0                  # no real movement → not J / Z
        p = p / max(float(p.sum()), 1e-9)
    i = int(p.argmax()); letter, prob = LETTERS[i], float(p[i])
    if rules and 'uv_split' in rules and letter in ('U', 'V'):
        prob = float(p[LETTERS.index('U')] + p[LETTERS.index('V')])
        letter = 'U' if finger_spread(f) < rules['uv_split'] else 'V'
    return letter, prob


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DATA FILES
# ─────────────────────────────────────────────────────────────────────────────

def count_letter_takes(data_dir: str) -> dict:
    return {L: len([f for f in os.listdir(os.path.join(data_dir, L)) if f.endswith('.npz')])
            if os.path.isdir(os.path.join(data_dir, L)) else 0 for L in LETTERS}


def save_letter_take(data_dir: str, letter: str, kp: np.ndarray, t: np.ndarray) -> str:
    folder = os.path.join(data_dir, letter)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.npz')
    np.savez_compressed(path, kp=kp.astype(np.float32), t=t.astype(np.float32))
    return path


def take_frame_gap(data_dirs: List[str]) -> Optional[float]:
    """Typical seconds between frames in the recorded takes."""
    gaps = []
    for d in data_dirs:
        for L in LETTERS:
            folder = os.path.join(d, L)
            if not os.path.isdir(folder):
                continue
            for fn in os.listdir(folder):
                if fn.endswith('.npz'):
                    with np.load(os.path.join(folder, fn)) as z:
                        t = z['t']
                    if len(t) > 2:
                        gaps.append(float(np.median(np.diff(t))))
    return float(np.median(gaps)) if gaps else None


def load_letter_windows(data_dirs: List[str], verbose: bool = True):
    """Returns list of (features, letter_index, take_id, signer)."""
    rows = []
    for d in data_dirs:
        signer = os.path.basename(os.path.normpath(d))
        n0 = len(rows)
        for li, L in enumerate(LETTERS):
            folder = os.path.join(d, L)
            if not os.path.isdir(folder):
                continue
            for fn in sorted(os.listdir(folder)):
                if not fn.endswith('.npz'):
                    continue
                with np.load(os.path.join(folder, fn)) as z:
                    kp, t = z['kp'], z['t']
                for f in take_windows(kp, t):
                    rows.append((f, li, f'{signer}/{L}/{fn}', signer))
        if verbose:
            print(f"  {d}: {len(rows) - n0} windows")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MODEL
# ─────────────────────────────────────────────────────────────────────────────

def make_model(n_classes: int = 26):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(FEAT_DIM, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
        nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
        nn.Linear(256, n_classes),
    )


class LetterModel:
    """Wraps the classifier with its input normalisation."""

    def __init__(self, net, mean: np.ndarray, std: np.ndarray, dt: Optional[float] = None,
                 rules: Optional[dict] = None):
        self.net, self.mean, self.std = net, mean, std
        self.rules = rules or {}
        self.dt = dt                     # seconds between frames in the training takes
        self.net.eval()

    def proba(self, feats: np.ndarray) -> np.ndarray:
        import torch
        x = torch.from_numpy(((feats - self.mean) / self.std).astype(np.float32))
        with torch.no_grad():
            return torch.softmax(self.net(x), -1).numpy()

    def save(self, path: str, info: dict) -> None:
        import torch
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({'model': self.net.state_dict(), 'mean': self.mean, 'std': self.std,
                    'letters': LETTERS, 'dt': self.dt, 'rules': self.rules, **info}, path)

    @staticmethod
    def load(path: str) -> 'LetterModel':
        import torch
        ck = torch.load(path, map_location='cpu', weights_only=False)
        net = make_model(len(ck['letters']))
        net.load_state_dict(ck['model'])
        return LetterModel(net, ck['mean'], ck['std'], ck.get('dt'), ck.get('rules'))


def train_letters(data_dirs: List[str], out: str, test_dirs: Optional[List[str]] = None,
                  epochs: int = 60, seed: int = 0, verbose: bool = True) -> dict:
    import torch
    import torch.nn as nn
    log = print if verbose else (lambda *a, **k: None)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    log("Letter data:")
    rows = load_letter_windows(data_dirs, verbose)
    if not rows:
        raise SystemExit(f"No letter takes found in {data_dirs}")
    # hold out one take per letter per signer (windows of one take stay together)
    takes = {}
    for f, y, take, signer in rows:
        takes.setdefault((signer, y), set()).add(take)
    val_takes = {sorted(v)[rng.integers(len(v))] for v in takes.values() if len(v) >= 2}
    tr = [(f, y) for f, y, take, _ in rows if take not in val_takes]
    va = [(f, y) for f, y, take, _ in rows if take in val_takes]
    log(f"{len(rows)} windows | train {len(tr)} | val {len(va)} (1 take per letter held out)")

    X = np.stack([f for f, _ in tr]); Y = np.array([y for _, y in tr], dtype=np.int64)
    mean, std = X.mean(0), X.std(0) + 1e-4
    net = make_model()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, best_state = -1.0, None

    def acc_on(items, model):
        if not items:
            return float('nan'), None
        P = model.proba(np.stack([f for f, _ in items]))
        yt = np.array([y for _, y in items]); yp = P.argmax(1)
        cm = np.zeros((26, 26), int)
        for a, b in zip(yt, yp):
            cm[a, b] += 1
        return float((yp == yt).mean()), cm

    for ep in range(1, epochs + 1):
        net.train()
        order = rng.permutation(len(X))
        for i in range(0, len(order), 128):
            idx = order[i:i + 128]
            xb = np.stack([augment(X[j], rng) for j in idx])
            xb = torch.from_numpy(((xb - mean) / std).astype(np.float32))
            loss = loss_fn(net(xb), torch.from_numpy(Y[idx]))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model = LetterModel(net, mean, std)
        acc, _ = acc_on(va, model) if va else acc_on(tr[:500], model)
        if acc >= best:                      # ties: keep the later, more confident model
            best = acc
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        net.train()
        if ep % 10 == 0 or ep == 1:
            log(f"epoch {ep:3d} | loss {loss.item():.3f} | val acc {100 * acc:5.1f}%")

    net.load_state_dict(best_state)
    dt = take_frame_gap(data_dirs)
    rules = learn_rules(X, Y)
    model = LetterModel(net, mean, std, dt, rules)
    if 'uv_split' in rules:
        mu, mv = rules['uv_medians']
        log(f"U/V check: fingertip gap U {mu:.2f}, V {mv:.2f} → split at {rules['uv_split']:.2f}")
    if va:
        Pv = model.proba(np.stack([f for f, _ in va]))
        plain = np.mean([LETTERS[int(p.argmax())] == LETTERS[y] for p, (_, y) in zip(Pv, va)])
        ruled = np.mean([apply_rules(rules, f, p)[0] == LETTERS[y] for p, (f, y) in zip(Pv, va)])
        log(f"Val accuracy: model {100 * plain:.1f}% | model + U/V and J/Z checks {100 * ruled:.1f}%")
    if dt:
        log(f"Recorded takes: {1 / dt:.1f} frames per second (live spelling will match this)")
    acc, cm = acc_on(va, model)
    log(f"\nBest val accuracy (per window): {100 * best:.1f}%")
    if cm is not None:
        weak = [(LETTERS[i], cm[i, i] / max(1, cm[i].sum()), LETTERS[int(np.argmax(np.where(np.arange(26) == i, -1, cm[i])))])
                for i in range(26) if cm[i].sum() > 0]
        weak = sorted(weak, key=lambda w: w[1])[:6]
        log("Weakest letters: " + ', '.join(f"{a} {100 * b:.0f}% (→{c})" for a, b, c in weak))
    test_acc = None
    if test_dirs:
        log("\nUnseen-signer letters:")
        trows = load_letter_windows(test_dirs, verbose)
        test_acc, tcm = acc_on([(f, y) for f, y, _, _ in trows], model)
        log(f"Unseen-signer accuracy (per window): {100 * test_acc:.1f}%")
    model.save(out, {'val_acc': best, 'sources': list(data_dirs)})
    log(f"Saved {out}")
    return {'val_acc': best, 'test_acc': test_acc}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — LIVE SPELLER
# ─────────────────────────────────────────────────────────────────────────────

class Speller:
    """
    Feed frames one by one. Returns a letter when one is accepted, 'END' when
    the hands go down after spelling, else None. Double letters (like the
    two A's in INAAYA) need a short break between them: move or bounce the
    hand slightly, as ASL signers do.
    """

    def __init__(self, model: LetterModel, stable: int = STABLE, min_prob: float = MIN_PROB):
        self.model, self.stable, self.min_prob = model, stable, min_prob
        self.reset()

    def reset(self) -> None:
        self.buf: List[np.ndarray] = []
        self.tbuf: List[Tuple[float, np.ndarray]] = []
        self.run_letter, self.run_len = None, 0
        self.last_letter, self.broken = None, True
        self.no_hand, self.spelled, self.ended = 0, '', False
        self.current = None                  # (letter, prob) of the latest window

    def _timed_window(self, kp: np.ndarray, t: float) -> Optional[List[np.ndarray]]:
        """
        WINDOW frames spaced like the training takes, whatever the live frame
        rate is. Without this, a slower or faster live loop changes the
        fingertip motion the model sees (and U can turn into V).
        """
        dt = self.model.dt
        span = dt * (WINDOW - 1)
        self.tbuf = [(tt, f) for tt, f in self.tbuf if t - tt <= span + 2 * dt] + [(t, kp)]
        if t - self.tbuf[0][0] < 0.8 * span:
            return None
        times = np.array([tt for tt, _ in self.tbuf])
        picks = [int(np.argmin(np.abs(times - (t - k * dt)))) for k in range(WINDOW - 1, -1, -1)]
        return [self.tbuf[i][1] for i in picks]

    def push(self, kp: np.ndarray, t: Optional[float] = None) -> Optional[str]:
        """kp: one frame (543, 3). t: its time in seconds (live), or None."""
        if signing_hand(kp) is None:
            self.no_hand += 1
            self.run_letter, self.run_len, self.broken = None, 0, True
            if self.no_hand >= END_FRAMES and self.spelled and not self.ended:
                self.ended = True                # emit END once; caller reads .spelled, then reset()
                return 'END'
            return None
        self.no_hand = 0
        if t is not None and self.model.dt:
            frames = self._timed_window(kp, t)
            if frames is None:
                return None
        else:
            self.buf = (self.buf + [kp])[-WINDOW:]
            if len(self.buf) < WINDOW:
                return None
            frames = self.buf
        f = window_features(frames)
        if f is None:
            return None
        p = self.model.proba(f[None])[0]
        letter, prob = apply_rules(self.model.rules, f, p)
        self.current = (letter, prob, finger_spread(f))
        if prob < self.min_prob:
            self.run_letter, self.run_len, self.broken = None, 0, True
            return None
        if letter == self.run_letter:
            self.run_len += 1
        else:
            if self.run_letter is not None:
                self.broken = True
            self.run_letter, self.run_len = letter, 1
        if self.run_len == self.stable and (letter != self.last_letter or self.broken):
            self.last_letter, self.broken = letter, False
            self.spelled += letter
            return letter
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — RECORD / LIVE APPS
# ─────────────────────────────────────────────────────────────────────────────

def record_letters(data_dir: str, takes: int, camera: int, only: Optional[List[str]]) -> None:
    import cv2
    import mediapipe as mp
    from record_phrases import pad_to_16x9, results_to_543, put

    letters = [L.upper() for L in only] if only else LETTERS
    bad = [L for L in letters if L not in LETTERS]
    if bad:
        sys.exit(f"Not letters: {bad}")
    counts = count_letter_takes(data_dir)
    queue = []
    for r in range(takes):
        rnd = [L for L in letters if counts[L] + sum(q == L for q in queue) < takes]
        random.shuffle(rnd); queue += rnd
    if not queue:
        print(f"All letters already have {takes} takes."); return
    print(f"Saving to {os.path.abspath(data_dir)} | {len(queue)} takes")

    mp_h, mp_draw = mp.solutions.holistic, mp.solutions.drawing_utils
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {camera}")
    state, buf, tb, t0, msg, msg_until, last = 'ready', [], [], 0.0, '', 0.0, None
    with mp_h.Holistic(model_complexity=1, refine_face_landmarks=False) as hol:
        while queue:
            ok, frame = cap.read()
            if not ok:
                break
            frame = pad_to_16x9(frame)
            res = hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            now = time.time()
            kp = results_to_543(res)
            L = queue[0]
            dur = MOTION_HOLD_S if L in MOTION_LETTERS else HOLD_S
            if state == 'rec':
                buf.append(kp); tb.append(now - t0)
                if now - t0 >= dur:
                    state = 'ready'
                    k, t = np.stack(buf), np.array(tb)
                    seen = sum(signing_hand(f) is not None for f in k)
                    if seen < 0.7 * len(k):
                        msg, msg_until = 'hand not seen well, not saved. Try again.', now + 3
                    else:
                        last = (save_letter_take(data_dir, L, k, t), L)
                        counts[L] += 1; queue.pop(0)
                        msg, msg_until = f'saved {L}', now + 1.5
            disp = frame.copy()
            for hand in (res.left_hand_landmarks, res.right_hand_landmarks):
                mp_draw.draw_landmarks(disp, hand, mp_h.HAND_CONNECTIONS)
            disp = cv2.flip(disp, 1)
            if queue:
                L = queue[0]
                put(disp, f'Letter:  {L}' + ('   (move it slowly 2 to 3 times)' if L in MOTION_LETTERS else '   (hold it steady)'),
                    50, (0, 255, 255), 1.1, 2)
                put(disp, f'take {counts[L] + 1}/{takes}   |   {len(queue)} left', 90)
            if state == 'rec':
                cv2.circle(disp, (disp.shape[1] - 40, 40), 15, (0, 0, 255), -1)
                put(disp, f'RECORDING {now - t0:.1f}s', disp.shape[0] - 25, (0, 0, 255))
            else:
                put(disp, 'make the letter, then SPACE   R = redo last   S = skip   Q = quit', disp.shape[0] - 25)
            if msg and now < msg_until:
                put(disp, msg, 130, (0, 200, 255))
            cv2.imshow('Sign2Text letters', disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' ') and state == 'ready':
                state, buf, tb, t0 = 'rec', [], [], time.time()
            elif key == ord('s') and state == 'ready':
                queue.append(queue.pop(0))
            elif key == ord('r') and state == 'ready' and last:
                os.remove(last[0]); counts[last[1]] -= 1; queue.insert(0, last[1]); last = None
                msg, msg_until = 'deleted last take', now + 2
    cap.release(); cv2.destroyAllWindows()
    show_stats(data_dir, takes)


def show_stats(data_dir: str, takes: int = 0) -> None:
    c = count_letter_takes(data_dir)
    print('  '.join(f"{L}:{c[L]}" for L in LETTERS))
    print(f"Total {sum(c.values())} takes" + (f" (target {takes * 26})" if takes else ''))


def live_letters(model_path: str, camera: int, min_prob: float = MIN_PROB) -> None:
    import cv2
    import mediapipe as mp
    from record_phrases import pad_to_16x9, results_to_543, put
    model = LetterModel.load(model_path)
    speller = Speller(model, min_prob=min_prob)
    mp_h, mp_draw = mp.solutions.holistic, mp.solutions.drawing_utils
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    words = []
    with mp_h.Holistic(model_complexity=1, refine_face_landmarks=False) as hol:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = pad_to_16x9(frame)
            res = hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ev = speller.push(results_to_543(res), time.time())
            if ev == 'END':
                words = (words + [speller.spelled])[-5:]; speller.reset()
            disp = frame.copy()
            for hand in (res.left_hand_landmarks, res.right_hand_landmarks):
                mp_draw.draw_landmarks(disp, hand, mp_h.HAND_CONNECTIONS)
            disp = cv2.flip(disp, 1)
            put(disp, f'Spelling: {speller.spelled}', 60, (0, 255, 255), 1.4, 3)
            if speller.current:
                put(disp, f'now: {speller.current[0]} {100 * speller.current[1]:.0f}%  gap {speller.current[2]:.2f}', 110, (200, 255, 200), 0.8)
            if words:
                put(disp, 'Words: ' + ' | '.join(words), disp.shape[0] - 60, (255, 255, 255), 0.8)
            put(disp, 'hold each letter still, lower hands to finish   C = clear   Q = quit', disp.shape[0] - 25, (180, 180, 180), 0.6)
            cv2.imshow('Sign2Text fingerspelling', disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                speller.reset(); words = []
    cap.release(); cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — SELF-TEST (fake hands)
# ─────────────────────────────────────────────────────────────────────────────

def _fake_letter_frame(letter_idx: int, rng, left: bool = False, jitter: float = 0.004,
                       phase: float = 0.0) -> np.ndarray:
    """A fake frame where each letter has its own fixed hand shape."""
    kp = np.zeros((543, 3), np.float32)
    kp[L_SH] = [0.42, 0.62, 0]; kp[R_SH] = [0.58, 0.62, 0]
    base = np.random.default_rng(100 + letter_idx).uniform(-0.04, 0.04, (21, 3)).astype(np.float32)
    base[0] = 0; base[9] = [0.0, -0.05, 0]
    hand = base + rng.normal(0, jitter, (21, 3)).astype(np.float32)
    wrist = np.array([0.55 + 0.02 * np.sin(phase), 0.45, 0], np.float32)
    pts = hand + wrist
    if left:                                              # mirror image, other slot
        pts[:, 0] = 2 * 0.5 - pts[:, 0]
        kp[LH0:LH0 + 21] = pts
    else:
        kp[RH0:RH0 + 21] = pts
    return kp


def selftest() -> None:
    import tempfile
    print("\n" + "=" * 60 + "\n  Fingerspelling self-test\n" + "=" * 60)
    rng = np.random.default_rng(0)

    print("── Test 1: left hand mirrored = right hand")
    fr = [_fake_letter_frame(3, rng, left=False, jitter=0) for _ in range(WINDOW)]
    fl = [_fake_letter_frame(3, rng, left=True, jitter=0) for _ in range(WINDOW)]
    a, b = window_features(fr), window_features(fl)
    assert a is not None and b is not None and np.allclose(a, b, atol=1e-5), np.abs(a - b).max()
    print("  ✓")

    print("── Test 2: missing hand → no features; higher hand is chosen")
    assert window_features([np.zeros((543, 3), np.float32)] * WINDOW) is None
    k = _fake_letter_frame(1, rng); k[LH0:LH0 + 21] = k[RH0:RH0 + 21] + [0, 0.3, 0]  # left lower
    assert signing_hand(k)[1] is False
    print("  ✓")

    print("── Test 3: record files, train, unseen-signer test, save, load")
    with tempfile.TemporaryDirectory() as d:
        letters = ['H', 'I', 'Y', 'U', 'M']
        for folder, left in (('a', False), ('b', True)):
            for L in letters:
                for _ in range(3):
                    n = 32
                    kp = np.stack([_fake_letter_frame(LETTERS.index(L), rng, left=left, phase=i / 5)
                                   for i in range(n)])
                    save_letter_take(os.path.join(d, folder), L, kp, np.arange(n) / 16.0)
        out = os.path.join(d, 'letters.pt')
        r = train_letters([os.path.join(d, 'a')], out, test_dirs=[os.path.join(d, 'b')],
                          epochs=120, verbose=False)
        assert r['val_acc'] > 0.9 and r['test_acc'] > 0.9, r
        model = LetterModel.load(out)
        print(f"  ✓ val {100 * r['val_acc']:.0f}%, left-handed unseen signer {100 * r['test_acc']:.0f}%")

        print("── Test 4: live speller spells HIYUMI (hold each letter, hands down to end)")
        sp = Speller(model)
        events = []
        for L in 'HIYUMI':
            for i in range(20):                                   # hold ~1.2 s
                e = sp.push(_fake_letter_frame(LETTERS.index(L), rng, phase=i / 5))
                if e: events.append(e)
        for _ in range(END_FRAMES + 1):
            e = sp.push(np.zeros((543, 3), np.float32))
            if e: events.append(e)
        assert ''.join(events[:-1]) == 'HIYUMI' and events[-1] == 'END', events
        print(f"  ✓ {''.join(events[:-1])} then END")

        print("── Test 4b: live window matches the training frame spacing at any live speed")
        assert model.dt is not None and abs(model.dt - 1 / 16) < 1e-6, model.dt
        for live_fps in (8, 16, 30):
            sp = Speller(model)
            for i in range(41):
                t = i / live_fps
                fr = _fake_letter_frame(LETTERS.index('U'), rng, jitter=0)
                fr[RH0:RH0 + 21, 0] += 0.05 * t                 # hand drifts right at a fixed speed
                w = sp._timed_window(fr, t)
            covered = (w[-1][RH0, 0] - w[0][RH0, 0]) / 0.05      # seconds the window spans
            assert abs(covered - (WINDOW - 1) / 16) <= 0.51 / live_fps + 1e-4, (live_fps, covered)
        print("  ✓ window covers the same time at 8, 16 and 30 fps")

        print("── Test 5: double letter needs a break (II only with a bounce)")
        sp.reset(); ev = []
        for i in range(30):
            e = sp.push(_fake_letter_frame(LETTERS.index('I'), rng, phase=i / 5))
            if e: ev.append(e)
        assert ev == ['I'], ev
        for _ in range(2):                                         # bounce: hand briefly lost
            sp.push(np.zeros((543, 3), np.float32))
        for i in range(20):
            e = sp.push(_fake_letter_frame(LETTERS.index('I'), rng, phase=i / 5))
            if e: ev.append(e)
        assert ev == ['I', 'I'], ev
        print("  ✓")

    print("── Test 6: U/V gap check and J/Z need real movement")
    feats, ys = [], []
    for L, gap, move in (('U', 0.05, 0), ('V', 0.40, 0), ('I', 0, 0.02), ('J', 0, 0.5)):
        for _ in range(30):
            f = rng.normal(0, 0.01, FEAT_DIM).astype(np.float32)
            f[3 * 12:3 * 12 + 2] = f[3 * 8:3 * 8 + 2] + [gap + rng.normal(0, 0.03), 0]
            k = 63 + 3 * TIPS.index(20); f[k] = move + abs(rng.normal(0, 0.01))
            feats.append(f); ys.append(LETTERS.index(L))
    rules = learn_rules(np.stack(feats), np.array(ys))
    assert 0.05 < rules['uv_split'] < 0.40, rules
    pv = np.zeros(26, np.float32); pv[LETTERS.index('V')] = 0.7; pv[LETTERS.index('U')] = 0.3
    assert apply_rules(rules, feats[0], pv)[0] == 'U'            # tips together → U, even if model says V
    assert apply_rules(rules, feats[30], pv)[0] == 'V'
    pj = np.zeros(26, np.float32); pj[LETTERS.index('J')] = 0.7; pj[LETTERS.index('I')] = 0.3
    assert apply_rules(rules, feats[60], pj)[0] == 'I'           # held still → I, not J
    assert apply_rules(rules, feats[90], pj)[0] == 'J'           # pinky moved → J
    print("  ✓")

    print("\nAll tests PASSED ✓")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['record', 'stats', 'train', 'live', 'selftest'])
    ap.add_argument('--data', nargs='+', default=['data_letters'])
    ap.add_argument('--test', nargs='+', default=None)
    ap.add_argument('--takes', type=int, default=4)
    ap.add_argument('--only', nargs='+', default=None, help='record only these letters')
    ap.add_argument('--model', default=os.path.join(ROOT, 'results', 'letters_model.pt'))
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--min-prob', type=float, default=MIN_PROB, help='live: confidence needed to accept a letter')
    a = ap.parse_args()
    if a.cmd == 'selftest':
        selftest()
    elif a.cmd == 'record':
        record_letters(a.data[0], a.takes, a.camera, a.only)
    elif a.cmd == 'stats':
        for d in a.data:
            print(d); show_stats(d, a.takes)
    elif a.cmd == 'train':
        train_letters(a.data, a.model, a.test)
    elif a.cmd == 'live':
        live_letters(a.model, a.camera, a.min_prob)
