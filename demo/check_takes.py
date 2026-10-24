"""
Check recorded takes before training: length, signing part, framing.

Run from the repo root:
    python demo/check_takes.py data_test
    python demo/check_takes.py data data_friend
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'training'))
from s2t_demo import load_takes, trim_active, active_frames, TARGET_FPS  # noqa: E402

ASPECT = 16 / 9


def shoulder_width(kp):
    ls, rs = kp[:, 11], kp[:, 12]
    ok = (np.abs(ls).sum(1) > 0) & (np.abs(rs).sum(1) > 0)
    if not ok.any():
        return None, 0.0
    w = np.hypot((ls[ok, 0] - rs[ok, 0]) * ASPECT, ls[ok, 1] - rs[ok, 1])
    return float(np.median(w)), float(ok.mean())


def main(dirs):
    takes, classes = load_takes(dirs, verbose=True)
    problems = 0
    print(f"\n{'phrase':22s} {'signer':12s} {'total s':>7s} {'signing s':>9s} {'shoulders':>9s}  note")
    for t in takes:
        kp = t['kp']
        name = classes[t['y']]
        total = len(kp) / TARGET_FPS
        sign = len(trim_active(kp)) / TARGET_FPS
        width, sh_seen = shoulder_width(kp)
        notes = []
        if name != '_idle' and active_frames(kp).sum() < 4:
            notes.append('NO SIGNING FOUND (hands never raised?)')
        if width is None or sh_seen < 0.8:
            notes.append('shoulders missing in many frames')
        elif width > 0.55:
            notes.append('very close to camera')
        if name != '_idle' and total > 5.5:
            notes.append('long take (did auto-stop work?)')
        problems += bool(notes)
        ws = f'{width:.2f}' if width is not None else '-'
        print(f"{name:22s} {t['signer'][:12]:12s} {total:7.1f} {sign:9.1f} {ws:>9s}  {'; '.join(notes)}")
    print(f"\n{len(takes)} takes, {problems} with notes.")
    print("Shoulders column = shoulder width in frame-height units. 0.20 to 0.45 is ideal, "
          "up to 0.55 is fine.")


if __name__ == '__main__':
    main(sys.argv[1:] or ['data'])
