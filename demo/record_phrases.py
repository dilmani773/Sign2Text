"""
Sign2Text Demo — Phrase Recorder
================================
Records ASL phrases from your webcam and saves MediaPipe Holistic keypoints
in the SAME format as How2Sign Holistic, so the trained encoder can reuse them.

    keypoints per frame (543, 3):
        [0:33] pose   [33:501] face   [501:522] left hand   [522:543] right hand
    missing parts = zeros

Setup (once, Python 3.10 / 3.11 / 3.12):
    pip install mediapipe==0.10.21 opencv-python numpy

Run:
    python record_phrases.py                 # record (30 takes per phrase)
    python record_phrases.py --takes 10      # fewer takes per session
    python record_phrases.py --stats         # show how many takes you have
    python record_phrases.py --selftest      # check the setup, no webcam needed

Keys while recording:
    SPACE   start signing / stop signing
    R       redo: delete the last saved take and record it again
    S       skip this phrase for now
    Q       quit (everything saved so far is kept)

Tips:
    - Same framing as How2Sign: whole upper body visible, hands never leave
      the frame, plain background, good light from the front.
    - Rest your hands down before pressing SPACE and after you finish.
    - Record over several days, in different clothes and lighting.
    - The "_idle" class is important: just sit, scratch your face, move a bit,
      without signing. It teaches the demo when NOT to output a phrase.

Output:
    data/<class_folder>/<timestamp>.npz   kp (T, 543, 3) float32, t (T,) seconds
    data/index.csv                        one row per take
"""

import os
os.environ.setdefault('GLOG_minloglevel', '2')        # hide MediaPipe info/warning spam
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import csv
import sys
import time
import random
import argparse
from datetime import datetime

import numpy as np

PHRASES = [
    "_idle",                    # no signing: resting, small movements
    "hello",
    "thank you",
    "please",
    "sorry",
    "yes",
    "no",
    "help me",
    "how are you",
    "i am fine",
    "what is your name",
    "nice to meet you",
    "good morning",
    "goodbye",
    "i don't understand",
    "again please",
    "I am good",
    "i understand",
    "bad",
    "i love you",
    "do you understand?",
]

N_POSE, N_FACE, N_HAND = 33, 468, 21
NUM_KP = N_POSE + N_FACE + 2 * N_HAND            # 543
LH0 = N_POSE + N_FACE                            # 501
RH0 = LH0 + N_HAND                               # 522

MIN_FRAMES   = 8       # shorter takes are rejected
MIN_HAND_PCT = 50.0    # warn if a hand is visible in fewer frames than this
MAX_SECONDS  = 6.0     # recording stops by itself after this


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers (tested by --selftest)
# ─────────────────────────────────────────────────────────────────────────────

def class_folder(i: int, phrase: str) -> str:
    slug = ''.join(c if c.isalnum() else '_' for c in phrase.lower()).strip('_')
    return f"{i:02d}_{slug or 'idle'}"


def pad_to_16x9(frame: np.ndarray) -> np.ndarray:
    """
    Pad (never crop) the frame to 16:9 with black borders, so x and y use
    the same scale as How2Sign (1280x720). Nothing in view is lost.
    """
    import cv2
    h, w = frame.shape[:2]
    target_w = int(round(h * 16 / 9))
    if w < target_w:
        left = (target_w - w) // 2
        return cv2.copyMakeBorder(frame, 0, 0, left, target_w - w - left, cv2.BORDER_CONSTANT)
    target_h = int(round(w * 9 / 16))
    if h < target_h:
        top = (target_h - h) // 2
        return cv2.copyMakeBorder(frame, top, target_h - h - top, 0, 0, cv2.BORDER_CONSTANT)
    return frame


def results_to_543(res) -> np.ndarray:
    """MediaPipe Holistic results → (543, 3) in How2Sign order, zeros if missing."""
    out = np.zeros((NUM_KP, 3), dtype=np.float32)

    def fill(lms, start, n):
        if lms is not None:
            out[start:start + n] = [[p.x, p.y, p.z] for p in lms.landmark[:n]]

    fill(res.pose_landmarks,       0,      N_POSE)
    fill(res.face_landmarks,       N_POSE, N_FACE)
    fill(res.left_hand_landmarks,  LH0,    N_HAND)
    fill(res.right_hand_landmarks, RH0,    N_HAND)
    return out


def hand_pct(kp: np.ndarray) -> float:
    """% of frames where at least one hand is detected."""
    lh = np.abs(kp[:, LH0:RH0]).sum(axis=(1, 2)) > 0
    rh = np.abs(kp[:, RH0:]).sum(axis=(1, 2)) > 0
    return float(100 * (lh | rh).mean()) if len(kp) else 0.0


def count_takes(data_dir: str) -> dict:
    counts = {}
    for i, p in enumerate(PHRASES):
        d = os.path.join(data_dir, class_folder(i, p))
        counts[i] = len([f for f in os.listdir(d) if f.endswith('.npz')]) if os.path.isdir(d) else 0
    return counts


def build_queue(counts: dict, takes: int, seed=None) -> list:
    """Round-robin in shuffled order, so each phrase is spread across the session."""
    rng = random.Random(seed)
    queue, need = [], {i: max(0, takes - c) for i, c in counts.items()}
    while any(need.values()):
        rnd = [i for i, n in need.items() if n > 0]
        rng.shuffle(rnd)
        for i in rnd:
            queue.append(i); need[i] -= 1
    return queue


def save_take(data_dir: str, cls: int, kp: np.ndarray, t: np.ndarray, session: str) -> str:
    folder = os.path.join(data_dir, class_folder(cls, PHRASES[cls]))
    os.makedirs(folder, exist_ok=True)
    name = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.npz'
    path = os.path.join(folder, name)
    np.savez_compressed(path, kp=kp.astype(np.float32), t=t.astype(np.float32))

    index = os.path.join(data_dir, 'index.csv')
    new = not os.path.exists(index)
    dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    with open(index, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['file', 'class_id', 'phrase', 'frames', 'seconds', 'fps', 'hand_pct', 'session'])
        w.writerow([os.path.relpath(path, data_dir), cls, PHRASES[cls], len(kp), round(dur, 3),
                    round((len(t) - 1) / dur, 1) if dur > 0 else 0, round(hand_pct(kp), 1), session])
    return path


def delete_take(data_dir: str, path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
    index = os.path.join(data_dir, 'index.csv')
    rel = os.path.relpath(path, data_dir)
    with open(index) as fh:
        rows = [r for r in csv.reader(fh) if r and r[0] != rel]
    with open(index, 'w', newline='') as fh:
        csv.writer(fh).writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Recorder
# ─────────────────────────────────────────────────────────────────────────────

def put(img, text, y, color=(255, 255, 255), scale=0.8, thick=2):
    import cv2
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def record(data_dir: str, takes: int, camera: int) -> None:
    import cv2
    import mediapipe as mp

    mp_h, mp_draw = mp.solutions.holistic, mp.solutions.drawing_utils
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {camera}. Try --camera 1")

    counts  = count_takes(data_dir)
    queue   = build_queue(counts, takes)
    session = datetime.now().strftime('%Y%m%d_%H%M')
    if not queue:
        print(f"All phrases already have {takes} takes. Use --takes with a higher number.")
        return
    print(f"{len(queue)} takes to record this session. Press SPACE to start each one.")

    state, buf_kp, buf_t, t0 = 'ready', [], [], 0.0
    last_saved, message, msg_until = None, '', 0.0
    fps_t, fps = time.time(), 0.0

    with mp_h.Holistic(model_complexity=1, refine_face_landmarks=False,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while queue:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed."); break
            frame = pad_to_16x9(frame)                         # NOT mirrored: keeps left/right correct
            res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - fps_t, 1e-3); fps_t = now

            if state == 'recording':
                buf_kp.append(results_to_543(res)); buf_t.append(now - t0)
                if now - t0 > MAX_SECONDS:
                    state = 'stop'

            # ── display (mirrored only for your eyes) ──
            disp = frame.copy()
            mp_draw.draw_landmarks(disp, res.pose_landmarks, mp_h.POSE_CONNECTIONS)
            mp_draw.draw_landmarks(disp, res.left_hand_landmarks, mp_h.HAND_CONNECTIONS)
            mp_draw.draw_landmarks(disp, res.right_hand_landmarks, mp_h.HAND_CONNECTIONS)
            disp = cv2.flip(disp, 1)

            cls = queue[0]
            done = counts[cls]
            put(disp, f'Sign:  "{PHRASES[cls]}"', 45, (0, 255, 255), 1.1, 2)
            put(disp, f'take {done + 1}/{takes}   |   {len(queue)} left this session   |   {fps:.0f} fps', 85)
            hands = ('L ' if res.left_hand_landmarks else '. ') + ('R' if res.right_hand_landmarks else '.')
            put(disp, f'hands seen: {hands}', 120)
            if state == 'ready':
                put(disp, 'SPACE = start    R = redo last    S = skip    Q = quit', disp.shape[0] - 25)
            elif state == 'recording':
                cv2.circle(disp, (disp.shape[1] - 40, 40), 15, (0, 0, 255), -1)
                put(disp, f'RECORDING {now - t0:.1f}s   SPACE = stop', disp.shape[0] - 25, (0, 0, 255))
            if message and now < msg_until:
                put(disp, message, 160, (0, 200, 255))
            cv2.imshow('Sign2Text recorder', disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                if state == 'ready':
                    state, buf_kp, buf_t, t0 = 'recording', [], [], time.time()
                elif state == 'recording':
                    state = 'stop'
            elif key == ord('s') and state == 'ready':
                queue.append(queue.pop(0))
                message, msg_until = 'skipped, will come back later', now + 2
            elif key == ord('r') and state == 'ready' and last_saved:
                path, c = last_saved
                delete_take(data_dir, path)
                counts[c] -= 1; queue.insert(0, c); last_saved = None
                message, msg_until = f'deleted last take of "{PHRASES[c]}", record it again', now + 3

            if state == 'stop':
                state = 'ready'
                kp, t = np.stack(buf_kp) if buf_kp else np.zeros((0, NUM_KP, 3)), np.array(buf_t)
                hp = hand_pct(kp)
                if len(kp) < MIN_FRAMES:
                    message, msg_until = 'too short, not saved. Try again.', now + 3
                    continue
                if cls != 0 and hp < MIN_HAND_PCT:
                    message, msg_until = f'hands seen in only {hp:.0f}% of frames, not saved. Try again.', now + 3
                    continue
                path = save_take(data_dir, cls, kp, t, session)
                counts[cls] += 1; queue.pop(0); last_saved = (path, cls)
                message, msg_until = f'saved ({len(kp)} frames, hands {hp:.0f}%)', now + 2

    cap.release()
    cv2.destroyAllWindows()
    print("\nSession done.")
    show_stats(data_dir, takes)


def show_stats(data_dir: str, takes: int) -> None:
    counts = count_takes(data_dir)
    total = sum(counts.values())
    print(f"\n{'phrase':28s} takes")
    for i, p in enumerate(PHRASES):
        bar = '#' * min(counts[i], 40)
        print(f"{p:28s} {counts[i]:3d}  {bar}")
    print(f"\nTotal: {total} takes (target {takes * len(PHRASES)})")


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (no webcam)
# ─────────────────────────────────────────────────────────────────────────────

def selftest() -> None:
    import tempfile
    import cv2
    import mediapipe as mp

    print("── Test 1: 16:9 padding")
    for h, w in [(480, 640), (720, 1280), (1080, 1440), (720, 1600)]:
        out = pad_to_16x9(np.zeros((h, w, 3), np.uint8))
        assert abs(out.shape[1] / out.shape[0] - 16 / 9) < 0.01, out.shape
    print("  4:3, 16:9, wide frames all → 16:9 ✓")

    print("── Test 2: MediaPipe Holistic runs, empty frame → all zeros (543, 3)")
    with mp.solutions.holistic.Holistic(model_complexity=1, refine_face_landmarks=False) as hol:
        res = hol.process(np.zeros((720, 1280, 3), np.uint8))
    kp = results_to_543(res)
    assert kp.shape == (543, 3) and not kp.any()
    print("  ✓")

    print("── Test 3: layout matches How2Sign")
    class P:                                     # fake landmark lists
        def __init__(s, n, v): s.landmark = [type('L', (), {'x': v, 'y': v, 'z': v})() for _ in range(n)]
    class R:
        pose_landmarks, face_landmarks = P(33, 1.0), P(468, 2.0)
        left_hand_landmarks, right_hand_landmarks = P(21, 3.0), P(21, 4.0)
    kp = results_to_543(R())
    assert (kp[0:33] == 1).all() and (kp[33:501] == 2).all()
    assert (kp[501:522] == 3).all() and (kp[522:543] == 4).all()
    print("  pose 0:33, face 33:501, LH 501:522, RH 522:543 ✓")

    print("── Test 4: save, index, redo, queue")
    with tempfile.TemporaryDirectory() as d:
        k = np.random.rand(30, 543, 3).astype(np.float32)
        t = np.linspace(0, 1, 30)
        p1 = save_take(d, 1, k, t, 's'); save_take(d, 1, k, t, 's')
        assert count_takes(d)[1] == 2
        delete_take(d, p1)
        assert count_takes(d)[1] == 1
        with open(os.path.join(d, 'index.csv')) as fh:          # close files: Windows
            rows = list(csv.reader(fh))                          # cannot delete open ones
        assert len(rows) == 2 and rows[1][2] == 'hello' and float(rows[1][5]) > 0
        with np.load(os.path.join(d, rows[1][0])) as z:
            assert z['kp'].shape == (30, 543, 3)
        q = build_queue(count_takes(d), takes=3, seed=0)
        assert q.count(1) == 2 and q.count(0) == 3 and len(q) == 3 * len(PHRASES) - 1
    print("  ✓")
    print("\nAll tests PASSED ✓  You can now run:  python record_phrases.py")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--takes', type=int, default=30, help='target takes per phrase')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--stats', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.stats:
        show_stats(a.data, a.takes)
    else:
        record(a.data, a.takes, a.camera)
        
        
        