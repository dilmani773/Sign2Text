"""
Sign2Text — Phase 1: Data Pipeline (sign → text)  v2
=====================================================
v2 changes (confirmed on real data):
    - Correct keypoint layout: pose [0:33], face [33:501],
      left hand [501:522], right hand [522:543]
    - x scaled by 16:9 aspect so x and y use the same units
    - Clips matched by SENTENCE_NAME (exact file name), no duplicates
    - Filter results cached, so the slow file scan runs once

Built on the SignX Phase 1 pipeline, changed for recognition.

Input  : How2Sign Holistic .npy, shape (T, 543, 3)
Output : features (T', 384) + English sentence

What changed from SignX:
    1. Keypoint subset      543 → 85 points (pose 13, hands 42, face 30)
    2. Missing parts        NaN/zero hands and face are masked, never smoothed
                            or normalized into fake coordinates
    3. Centering            shoulder midpoint (not hips, which are often
                            out of frame in How2Sign)
    4. Scale                median shoulder width per clip (stable)
    5. Long clips           uniformly resampled to max_frames, never cut,
                            so the whole sentence stays in the clip
    6. Hand-local features  hand coords relative to wrist, for handshape
    7. Presence flags       3 per-frame flags: left hand, right hand, face
    8. Smoothing            off by default (can blur fingerspelling),
                            masked when on
    9. Short / corrupt clips are dropped at startup, not replaced with zeros
   10. Optional augmentation for training

Feature layout per frame (384):
    [0   : 255]  85 selected keypoints × xyz, body-normalized
    [255 : 381]  42 hand keypoints × xyz, wrist-relative
    [381 : 384]  presence flags (left hand, right hand, face)

Kaggle paths:
    DATA_ROOT = '/kaggle/input/datasets/psewmuthu/how2sign-holistic/how2sign_holistic_features'
    TRAIN_CSV = DATA_ROOT + '/metadata/how2sign_realigned_train.csv'
    TRAIN_NPY = DATA_ROOT + '/train/frontal/'

Run locally:
    python s2t_phase1_dataset.py
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
# How2Sign Holistic order (543), confirmed by plotting + wrist z = 0 check:
#   [0 : 33] pose   [33 : 501] face   [501 : 522] left hand   [522 : 543] right hand

NUM_KEYPOINTS = 543
COORDS        = 3
FILE_SUFFIX   = '_holistic.npy'          # file = SENTENCE_NAME + FILE_SUFFIX
ASPECT        = 1280 / 720               # How2Sign video is 16:9; x *= ASPECT

FACE_OFFSET = 33

# Pose: nose + shoulders, elbows, wrists, and the small hand points on the pose
POSE_KEEP = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

# Face mesh: outer lips (20) + eyebrows (10). Mouth and brows carry ASL grammar.
LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
              291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
BROWS      = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
FACE_KEEP  = [FACE_OFFSET + i for i in LIPS_OUTER + BROWS]

LH_KEEP = list(range(501, 522))
RH_KEEP = list(range(522, 543))

KEEP_IDX = POSE_KEEP + LH_KEEP + RH_KEEP + FACE_KEEP
N_KEEP   = len(KEEP_IDX)                               # 85

# Slices inside the 85-point array
POSE_SL = slice(0, 13)
LH_SL   = slice(13, 34)
RH_SL   = slice(34, 55)
FACE_SL = slice(55, 85)

L_SHOULDER = POSE_KEEP.index(11)
R_SHOULDER = POSE_KEEP.index(12)

# Inside a 21-point hand: 0 = wrist, 9 = middle finger base
HAND_WRIST, HAND_MID_MCP = 0, 9

FEATURE_DIM = N_KEEP * COORDS + 42 * COORDS + 3        # 384


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FILENAME LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def build_npy_lookup(npy_dir: str) -> Dict[str, str]:
    """Scan npy_dir → { SENTENCE_NAME: full_path }. Names are unique."""
    lookup = {
        fname[: -len(FILE_SUFFIX)]: os.path.join(npy_dir, fname)
        for fname in os.listdir(npy_dir) if fname.endswith(FILE_SUFFIX)
    }
    log.info(f"NPY lookup: {len(lookup)} files in '{npy_dir}'")
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def select_and_mask(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    (T, 543, 3) → x (T, 85, 3) with no NaN, present (T, 85) bool.
    A point counts as missing if it is NaN or all-zero.
    Hands and face are marked present/missing as a whole part per frame,
    because MediaPipe drops the full hand or face, not single points.
    """
    x   = raw[:, KEEP_IDX, :].astype(np.float32)
    nan = np.isnan(x).any(axis=2)                       # (T, 85)
    x   = np.nan_to_num(x, nan=0.0)
    present = (~nan) & (np.abs(x).sum(axis=2) > 0)      # (T, 85)
    x[..., 0] *= ASPECT                                 # square units

    for sl in (LH_SL, RH_SL, FACE_SL):
        part_ok = present[:, sl].mean(axis=1) > 0.5     # (T,)
        present[:, sl] = part_ok[:, None]

    x[~present] = 0.0
    return x, present


def resample_time(x: np.ndarray, present: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pick n evenly spaced frames (nearest index). Keeps masks exact."""
    idx = np.linspace(0, x.shape[0] - 1, n).round().astype(int)
    return x[idx], present[idx]


def masked_smooth(x: np.ndarray, present: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian smoothing over time that ignores missing frames.
    Missing values do not leak into real ones, and stay 0.
    """
    from scipy.ndimage import gaussian_filter1d
    w   = present.astype(np.float32)                    # (T, K)
    num = gaussian_filter1d(x * w[..., None], sigma=sigma, axis=0)
    den = gaussian_filter1d(w, sigma=sigma, axis=0)[..., None]
    out = np.where(den > 1e-3, num / np.maximum(den, 1e-6), 0.0)
    return (out * w[..., None]).astype(np.float32)


def normalize_body(x: np.ndarray, present: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, bool]:
    """
    Center on the shoulder midpoint (per frame), scale by the clip's
    median shoulder width. Frames with no shoulders use the median center.
    Returns (x_norm, ok). ok=False if the clip has no usable shoulders.
    """
    ls, rs  = x[:, L_SHOULDER], x[:, R_SHOULDER]
    pose_ok = present[:, L_SHOULDER] & present[:, R_SHOULDER]
    if not pose_ok.any():
        return x, False

    center = (ls + rs) / 2.0                            # (T, 3)
    center[~pose_ok] = np.median(center[pose_ok], axis=0)

    width = np.linalg.norm(ls[pose_ok] - rs[pose_ok], axis=1)
    scale = max(float(np.median(width)), eps)

    out = (x - center[:, None, :]) / scale
    out[~present] = 0.0
    return out.astype(np.float32), True


def hand_local(x: np.ndarray, present: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Wrist-relative hand coordinates, scaled by wrist → middle-finger-base
    length. Captures handshape independent of where the hand is.
    Returns (T, 42, 3).
    """
    outs = []
    for sl in (LH_SL, RH_SL):
        h    = x[:, sl]                                 # (T, 21, 3)
        ok   = present[:, sl][:, 0]                     # (T,)
        loc  = h - h[:, HAND_WRIST:HAND_WRIST + 1]
        size = np.linalg.norm(h[:, HAND_MID_MCP] - h[:, HAND_WRIST], axis=1)
        loc  = loc / np.maximum(size, eps)[:, None, None]
        loc[~ok] = 0.0
        outs.append(loc)
    return np.concatenate(outs, axis=1).astype(np.float32)


def augment(x: np.ndarray, present: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light spatial augmentation on normalized coords: scale, xy-rotation, shift."""
    s     = rng.uniform(0.9, 1.1)
    theta = np.deg2rad(rng.uniform(-10, 10))
    c, sn = np.cos(theta), np.sin(theta)
    rot   = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]], dtype=np.float32)
    shift = rng.uniform(-0.05, 0.05, size=3).astype(np.float32)
    shift[2] = 0.0
    out = (x @ rot.T) * s + shift
    out[~present] = 0.0
    return out.astype(np.float32)


def select_stride(raw: np.ndarray, stride: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Stage 1 (cacheable): (T, 543, 3) → x (T', 85, 3), present (T', 85)."""
    x, present = select_and_mask(raw)
    return x[::stride], present[::stride]


def build_features(
    raw:          np.ndarray,
    max_frames:   int,
    stride:       int                         = 2,
    smooth_sigma: float                       = 0.0,
    rng:          Optional[np.random.Generator] = None,
) -> Optional[np.ndarray]:
    """
    Full preprocessing: (T, 543, 3) → (T', 384). Returns None if unusable.
    rng=None means no augmentation (val/test/inference).
    """
    x, present = select_stride(raw, stride)
    return features_from_selected(x, present, max_frames, smooth_sigma, rng)


def features_from_selected(
    x:            np.ndarray,
    present:      np.ndarray,
    max_frames:   int,
    smooth_sigma: float                       = 0.0,
    rng:          Optional[np.random.Generator] = None,
) -> Optional[np.ndarray]:
    """Stage 2 (per epoch): selected + strided points → (T', 384) features."""
    x       = x.astype(np.float32)          # copy (preload arrays stay untouched)
    present = present.copy()

    # Time: speed augmentation, then cap length by resampling
    T = x.shape[0]
    if rng is not None:
        T = max(2, int(round(T * rng.uniform(0.8, 1.2))))
    T = min(T, max_frames)
    if T != x.shape[0]:
        x, present = resample_time(x, present, T)

    if smooth_sigma > 0:
        x = masked_smooth(x, present, smooth_sigma)

    x, ok = normalize_body(x, present)
    if not ok:
        return None

    if rng is not None:
        x = augment(x, present, rng)

    local = hand_local(x, present)
    flags = np.stack([
        present[:, LH_SL.start],
        present[:, RH_SL.start],
        present[:, FACE_SL.start],
    ], axis=1).astype(np.float32)

    feats = np.concatenate([
        x.reshape(T, -1),
        local.reshape(T, -1),
        flags,
    ], axis=1)
    return feats.astype(np.float32)                     # (T, 384)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — DATASET
# ─────────────────────────────────────────────────────────────────────────────

class How2SignS2T(Dataset):
    """
    One item = { 'sid': str, 'text': str, 'features': FloatTensor (T, 384) }
    """

    def __init__(
        self,
        csv_path:     str,
        npy_dir:      str,
        max_frames:   int   = 256,     # after stride
        min_frames:   int   = 10,      # raw frames
        stride:       int   = 2,
        smooth_sigma: float = 0.0,
        augment:      bool  = False,
        seed:         int   = 42,
        cache_path:   Optional[str] = None,   # e.g. '/kaggle/working/train_index.csv'
    ) -> None:
        super().__init__()
        self.max_frames   = max_frames
        self.min_frames   = min_frames
        self.stride       = stride
        self.smooth_sigma = smooth_sigma
        self.augment      = augment
        self.pre          = None            # filled by preload()

        self.npy_lookup = build_npy_lookup(npy_dir)
        if cache_path and os.path.exists(cache_path):
            self.df = pd.read_csv(cache_path, keep_default_na=False)
            self.df = self.df[self.df['_sid'].isin(self.npy_lookup)].reset_index(drop=True)
            log.info(f"Loaded cached index: {cache_path}")
        else:
            self.df = self._load_csv(csv_path)
            self.df = self._filter(self.df)
            if cache_path:
                self.df[['_sid', '_text', '_len']].to_csv(cache_path, index=False)
                log.info(f"Saved index cache: {cache_path}")

        log.info(
            f"Dataset ready: {len(self.df)} clips | stride={stride} | "
            f"max_frames={max_frames} | smooth={smooth_sigma} | augment={augment}"
        )

    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(csv_path, sep='\t')
            if df.shape[1] < 2:
                raise ValueError
        except Exception:
            log.warning("TSV parse failed, retrying with comma.")
            df = pd.read_csv(csv_path)

        df.columns = [c.strip().upper() for c in df.columns]
        # SENTENCE_NAME matches the file name exactly; SENTENCE_ID does not
        self.id_col = next((c for c in ['SENTENCE_NAME', 'SENTENCE_ID', 'ID']
                            if c in df.columns), None)
        self.text_col = next((c for c in ['SENTENCE', 'ENGLISH_SENTENCE', 'TEXT', 'TRANSLATION']
                              if c in df.columns), None)
        if not self.id_col or not self.text_col:
            raise KeyError(f"Missing ID or text column. Got: {list(df.columns)}")

        df = df.dropna(subset=[self.id_col, self.text_col]).copy()
        df['_sid']  = df[self.id_col].astype(str).str.strip()
        df['_text'] = df[self.text_col].astype(str).str.strip()
        return df[df['_text'] != ''].reset_index(drop=True)

    def _filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop clips with no file, bad shape, or too few frames (header scan only)."""
        df = df[df['_sid'].isin(self.npy_lookup)].copy()
        lengths, keep = [], []
        bad_shape = too_short = 0
        for sid in df['_sid']:
            try:
                arr = np.load(self.npy_lookup[sid], mmap_mode='r', allow_pickle=True)
                if arr.ndim != 3 or arr.shape[1:] != (NUM_KEYPOINTS, COORDS):
                    bad_shape += 1; keep.append(False); lengths.append(0); continue
                if arr.shape[0] < self.min_frames:
                    too_short += 1; keep.append(False); lengths.append(arr.shape[0]); continue
                keep.append(True); lengths.append(arr.shape[0])
            except Exception:
                bad_shape += 1; keep.append(False); lengths.append(0)
        df['_len'] = lengths
        df = df[keep].reset_index(drop=True)
        log.info(f"Filter: dropped {bad_shape} bad/unreadable, {too_short} too short.")
        return df

    def preload(self, cache_file: Optional[str] = None, workers: int = 8) -> None:
        """
        Read every clip once, keep only the 85 points (float32), hold in RAM.
        Call AFTER clean_index so dropped clips are not loaded.
        cache_file: .npz path to load from if it exists, else save to.
        """
        sids = self.df['_sid'].tolist()
        if cache_file and os.path.exists(cache_file):
            z = np.load(cache_file)
            index = {s: i for i, s in enumerate(z['sids'].tolist())}
            if all(s in index for s in sids):
                self.pre = (z['x'], z['p'], z['offsets'], index)
                log.info(f"Preload: loaded {len(sids)} clips from {cache_file}")
                return
            log.warning("Preload cache does not cover this dataset, rebuilding.")

        from concurrent.futures import ThreadPoolExecutor

        def load(sid):
            raw = np.load(self.npy_lookup[sid], allow_pickle=True)
            x, p = select_stride(raw.astype(np.float32), self.stride)
            return x.astype(np.float32), p

        xs, ps = [], []
        with ThreadPoolExecutor(workers) as ex:
            for i, (x, p) in enumerate(ex.map(load, sids)):
                xs.append(x); ps.append(p)
                if (i + 1) % 5000 == 0:
                    log.info(f"Preload: {i + 1}/{len(sids)}")
        offsets = np.zeros(len(xs) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(x) for x in xs])
        X, P = np.concatenate(xs), np.concatenate(ps)
        index = {s: i for i, s in enumerate(sids)}
        self.pre = (X, P, offsets, index)
        log.info(f"Preload: {len(sids)} clips, {X.nbytes / 1e9:.2f} GB in RAM")
        if cache_file:
            np.savez(cache_file, x=X, p=P, offsets=offsets, sids=np.array(sids))
            log.info(f"Preload: saved {cache_file}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        # Fresh random stream per call: differs across workers and epochs
        rng = np.random.default_rng() if self.augment else None
        if self.pre is not None:
            X, P, off, index = self.pre
            i = index[row['_sid']]
            feats = features_from_selected(X[off[i]:off[i + 1]], P[off[i]:off[i + 1]],
                                           self.max_frames, self.smooth_sigma, rng)
        else:
            raw = np.load(self.npy_lookup[row['_sid']], allow_pickle=True).astype(np.float32)
            feats = build_features(raw, self.max_frames, self.stride, self.smooth_sigma, rng)
        if feats is None:
            # No shoulders anywhere: fall back to a neighbour instead of a fake clip
            log.warning(f"No usable pose in {row['_sid']}, using next clip.")
            return self.__getitem__((idx + 1) % len(self))
        return {'sid': row['_sid'], 'text': row['_text'], 'features': torch.from_numpy(feats)}

    def frame_stats(self) -> Dict[str, float]:
        L = self.df['_len'].to_numpy()
        return {
            'count': len(L), 'min': int(L.min()), 'max': int(L.max()),
            'mean': round(float(L.mean()), 1), 'median': float(np.median(L)),
            'p95': float(np.percentile(L, 95)),
            'pct_resampled': round(100 * float((L / self.stride > self.max_frames).mean()), 2),
        }


def clean_index(ds: 'How2SignS2T', min_fpw: float = 3, max_fpw: float = 40,
                max_words: int = 40, max_raw_frames: int = 900) -> None:
    """
    Drop bad pairs in place (decided in Phase 2 EDA):
        frames per word < 3   → misaligned (video too short for the sentence)
        frames per word > 40  → misaligned the other way
        words > 40            → would be truncated by the decoder
        raw frames > 900      → too squashed by resampling
    """
    d   = ds.df
    w   = d['_text'].str.split().str.len()
    fpw = d['_len'] / w
    rules = {
        'misaligned (fpw < min)': fpw < min_fpw,
        'too slow (fpw > max)':   fpw > max_fpw,
        'too many words':         w > max_words,
        'too long':               d['_len'] > max_raw_frames,
    }
    drop = np.zeros(len(d), dtype=bool)
    for name, r in rules.items():
        log.info(f"clean: {name:24s} {int(r.sum())}")
        drop |= r.to_numpy()
    ds.df = d[~drop].reset_index(drop=True)
    log.info(f"clean: kept {len(ds.df)} of {len(d)} ({100 * len(ds.df) / len(d):.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — COLLATOR
# ─────────────────────────────────────────────────────────────────────────────

class Collator:
    """
    Pads features and (optionally) tokenizes text for the decoder.

    Returns:
        sids, texts   : List[str]
        features      : FloatTensor (B, max_T, 384), zero padded
        feature_mask  : BoolTensor  (B, max_T), True = real frame
        lengths       : LongTensor  (B,)
        labels        : LongTensor  (B, L), pad = -100   (if tokenizer given)
    """

    def __init__(self, tokenizer=None, max_text_len: int = 64) -> None:
        self.tokenizer    = tokenizer
        self.max_text_len = max_text_len

    def __call__(self, batch: List[Dict]) -> Dict:
        feats   = [b['features'] for b in batch]
        B       = len(feats)
        lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
        max_T   = int(lengths.max())

        padded = torch.zeros(B, max_T, FEATURE_DIM, dtype=torch.float32)
        mask   = torch.zeros(B, max_T, dtype=torch.bool)
        for i, (f, L) in enumerate(zip(feats, lengths.tolist())):
            padded[i, :L] = f
            mask[i, :L]   = True

        out = {
            'sids':         [b['sid'] for b in batch],
            'texts':        [b['text'] for b in batch],
            'features':     padded,
            'feature_mask': mask,
            'lengths':      lengths,
        }
        if self.tokenizer is not None:
            enc = self.tokenizer(out['texts'], padding=True, truncation=True,
                                 max_length=self.max_text_len, return_tensors='pt')
            labels = enc['input_ids'].clone()
            labels[enc['attention_mask'] == 0] = -100
            out['labels'] = labels
        return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloader(
    csv_path:    str,
    npy_dir:     str,
    tokenizer                  = None,
    batch_size:  int           = 16,
    shuffle:     bool          = True,
    augment:     bool          = False,
    num_workers: int           = 2,
    **ds_kwargs,
) -> Tuple[DataLoader, How2SignS2T]:
    dataset = How2SignS2T(csv_path, npy_dir, augment=augment, **ds_kwargs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=Collator(tokenizer),
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )
    return loader, dataset


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

def _fake_clip(rng, T, lh_mode='ok', rh_mode='ok'):
    """Realistic-ish dummy clip. Hand modes: ok | zero | nan | half (missing half the frames)."""
    m = rng.uniform(0.3, 0.7, size=(T, NUM_KEYPOINTS, COORDS)).astype(np.float32)
    m[:, 11] = [0.40, 0.40, 0.0]                        # left shoulder
    m[:, 12] = [0.60, 0.40, 0.0]                        # right shoulder
    for sl, mode in ((slice(501, 522), lh_mode), (slice(522, 543), rh_mode)):
        if mode == 'zero':
            m[:, sl] = 0.0
        elif mode == 'nan':
            m[:, sl] = np.nan
        elif mode == 'half':
            m[: T // 2, sl] = 0.0
    return m


if __name__ == '__main__':
    import tempfile

    print("\n" + "=" * 65)
    print("  Sign2Text Phase 1 — Smoke Test")
    print("=" * 65 + "\n")
    rng = np.random.default_rng(0)

    with tempfile.TemporaryDirectory() as tmp:
        npy_dir, meta = os.path.join(tmp, 'frontal'), os.path.join(tmp, 'meta')
        os.makedirs(npy_dir); os.makedirs(meta)

        rows, modes = [], ['ok', 'zero', 'nan', 'half']
        for i in range(12):
            sid = f'clip_{i:03d}-{i}-rgb_front'
            T   = 900 if i == 0 else int(rng.integers(40, 300))   # clip 0 is very long
            clip = _fake_clip(rng, T, lh_mode=modes[i % 4], rh_mode='ok')
            np.save(os.path.join(npy_dir, f'{sid}{FILE_SUFFIX}'), clip)
            rows.append({'SENTENCE_NAME': sid, 'SENTENCE': f'Test sentence number {i}.'})

        np.save(os.path.join(npy_dir, f'short-1-rgb_front{FILE_SUFFIX}'), _fake_clip(rng, 5))
        rows.append({'SENTENCE_NAME': 'short-1-rgb_front', 'SENTENCE': 'Too short.'})
        rows.append({'SENTENCE_NAME': 'missing-1-rgb_front', 'SENTENCE': 'No file.'})
        csv = os.path.join(meta, 'train.csv')
        pd.DataFrame(rows).to_csv(csv, sep='\t', index=False)

        print("── Test 1: Filtering")
        ds = How2SignS2T(csv, npy_dir, max_frames=256, stride=2)
        assert len(ds) == 12, len(ds)
        print("  12 kept, short and missing dropped ✓")

        print("── Test 2: Shape, no NaN, length cap")
        for i in range(len(ds)):
            f = ds[i]['features']
            assert f.shape[1] == FEATURE_DIM and f.shape[0] <= 256
            assert torch.isfinite(f).all()
        print(f"  feature dim {FEATURE_DIM}, all finite, long clip → {ds[0]['features'].shape[0]} frames ✓")

        print("── Test 3: Missing hands stay zero, flags correct")
        f_zero = ds[1]['features'].numpy()              # left hand zero
        f_nan  = ds[2]['features'].numpy()              # left hand NaN
        lh_cols = slice(LH_SL.start * 3, LH_SL.stop * 3)
        for f in (f_zero, f_nan):
            assert np.all(f[:, lh_cols] == 0) and np.all(f[:, -3] == 0) and np.all(f[:, -2] == 1)
        f_half = ds[3]['features'].numpy()
        assert f_half[0, -3] == 0 and f_half[-1, -3] == 1
        print("  zero/NaN hands masked, half-missing flags switch mid-clip ✓")

        print("── Test 4: Normalization")
        x = ds[4]['features'].numpy()[:, : N_KEEP * 3].reshape(-1, N_KEEP, 3)
        mid = (x[:, L_SHOULDER] + x[:, R_SHOULDER]) / 2
        w   = np.linalg.norm(x[:, L_SHOULDER] - x[:, R_SHOULDER], axis=1)
        assert np.abs(mid).max() < 1e-4 and np.allclose(w, 1.0, atol=1e-4)
        print("  shoulder midpoint = 0, shoulder width = 1 ✓")

        print("── Test 5: Masked smoothing does not leak into missing frames")
        ds_s = How2SignS2T(csv, npy_dir, smooth_sigma=1.5)
        f = ds_s[3]['features'].numpy()
        missing = f[:, -3] == 0
        assert np.all(f[missing][:, lh_cols] == 0)
        print("  ✓")

        print("── Test 6: Augmentation + collator with tokenizer")
        class FakeTok:
            def __call__(self, texts, padding, truncation, max_length, return_tensors):
                ids = [[5] * (len(t.split()) + 1) for t in texts]
                L = max(map(len, ids))
                ii = torch.tensor([r + [0] * (L - len(r)) for r in ids])
                return {'input_ids': ii, 'attention_mask': (ii != 0).long()}
        loader, _ = build_dataloader(csv, npy_dir, tokenizer=FakeTok(), batch_size=4,
                                     augment=True, num_workers=0)
        b = next(iter(loader))
        assert b['features'].shape[0] == 4 and b['features'].shape[2] == FEATURE_DIM
        assert (b['labels'] == -100).any() or b['labels'].shape[1] > 0
        for i, L in enumerate(b['lengths'].tolist()):
            assert b['feature_mask'][i, :L].all() and not b['feature_mask'][i, L:].any()
        print(f"  batch features {tuple(b['features'].shape)}, labels {tuple(b['labels'].shape)} ✓")

        print("── Test 7: Cache")
        cache = os.path.join(tmp, 'index.csv')
        d1 = How2SignS2T(csv, npy_dir, cache_path=cache)
        d2 = How2SignS2T(csv, npy_dir, cache_path=cache)
        assert len(d1) == len(d2) == 12 and d2[5]['text'] == d1[5]['text']
        print("  cache saved and reloaded ✓")

        print("── Test 8: Preload gives the same features as disk")
        d = How2SignS2T(csv, npy_dir)
        disk = [d[i]['features'] for i in range(len(d))]
        pre_file = os.path.join(tmp, 'pre.npz')
        d.preload(pre_file)
        d2 = How2SignS2T(csv, npy_dir); d2.preload(pre_file)
        for i in range(len(d)):
            assert torch.allclose(disk[i], d2[i]['features'], atol=1e-5)
        print("  preload built, saved, reloaded, matches disk ✓")

        print("── Test 9: Stats")
        for k, v in ds.frame_stats().items():
            print(f"  {k:14s}: {v}")

    print("\n" + "=" * 65)
    print("  All tests PASSED ✓")
    print("=" * 65)
    print("""
── Kaggle usage ──────────────────────────────────────────────────────
DATA_ROOT = '/kaggle/input/datasets/psewmuthu/how2sign-holistic/how2sign_holistic_features'
TRAIN_CSV = DATA_ROOT + '/metadata/how2sign_realigned_train.csv'
TRAIN_NPY = DATA_ROOT + '/train/frontal/'
VAL_CSV   = DATA_ROOT + '/metadata/how2sign_realigned_val.csv'
VAL_NPY   = DATA_ROOT + '/val/frontal/'

train_loader, train_ds = build_dataloader(TRAIN_CSV, TRAIN_NPY, augment=True,
                                          cache_path='/kaggle/working/train_index.csv')
val_loader,   val_ds   = build_dataloader(VAL_CSV, VAL_NPY, shuffle=False,
                                          cache_path='/kaggle/working/val_index.csv')
print(train_ds.frame_stats())
""")
