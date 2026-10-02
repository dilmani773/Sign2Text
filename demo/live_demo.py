"""
Sign2Text — Live webcam demo
============================
Sign a phrase, lower your hands, and the phrase appears on screen.

How it works:
    webcam → MediaPipe Holistic → (543, 3) per frame
    → Segmenter: starts when a hand is raised, ends when hands go down
    → same features as training → phrase classifier → text on screen

Run from the repo root (after training/s2t_demo.py made the model):
    python demo/live_demo.py
    python demo/live_demo.py --model results/demo_model.pt --threshold 0.5

Keys:  C = clear history    Q = quit
"""

import os
os.environ.setdefault('GLOG_minloglevel', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import sys
import time
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'training'))
sys.path.insert(0, os.path.join(ROOT, 'demo'))


def main(model_path: str, threshold: float, camera: int) -> None:
    import cv2
    import torch
    import mediapipe as mp
    from s2t_demo import PhraseClassifier, Segmenter, clip_features, active_frames
    from record_phrases import pad_to_16x9, results_to_543, put

    ck = torch.load(model_path, map_location='cpu', weights_only=False)
    classes = ck['classes']
    model = PhraseClassifier(len(classes), ck['encoder_cfg'])
    model.load_state_dict(ck['model'])
    model.eval()
    print(f"Loaded {model_path}: {len(classes)} phrases, val acc {100 * ck['val_acc']:.0f}%")

    mp_h, mp_draw = mp.solutions.holistic, mp.solutions.drawing_utils
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {camera}. Try --camera 1")

    seg = Segmenter()
    history, last, last_until = [], None, 0.0
    fps, fps_t = 0.0, time.time()

    with mp_h.Holistic(model_complexity=1, refine_face_landmarks=False,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = pad_to_16x9(frame)
            res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - fps_t, 1e-3); fps_t = now
            kp = results_to_543(res)

            segment = seg.push(kp, now)
            if segment is not None:
                feats = torch.from_numpy(clip_features(segment))[None]
                mask = torch.ones(1, feats.shape[1], dtype=torch.bool)
                with torch.no_grad():
                    prob = torch.softmax(model(feats, mask), -1)[0]
                p, i = prob.max(0)
                name = classes[int(i)]
                if name == '_idle':
                    last = None
                elif p.item() >= threshold:
                    last = (name, p.item())
                    history = (history + [name])[-6:]
                else:
                    last = ('?', p.item())
                last_until = now + 3.0

            disp = frame.copy()
            mp_draw.draw_landmarks(disp, res.pose_landmarks, mp_h.POSE_CONNECTIONS)
            mp_draw.draw_landmarks(disp, res.left_hand_landmarks, mp_h.HAND_CONNECTIONS)
            mp_draw.draw_landmarks(disp, res.right_hand_landmarks, mp_h.HAND_CONNECTIONS)
            disp = cv2.flip(disp, 1)

            signing = bool(seg.seg)
            put(disp, 'signing...' if signing else 'ready: raise your hands to sign', 45,
                (0, 0, 255) if signing else (200, 200, 200), 0.9)
            if last and now < last_until:
                text = last[0] if last[0] == '?' else f'{last[0]}  ({100 * last[1]:.0f}%)'
                put(disp, text, 110, (0, 255, 255), 1.6, 3)
            if history:
                put(disp, ' | '.join(history), disp.shape[0] - 70, (255, 255, 255), 0.8)
            put(disp, f'C = clear   Q = quit   |   {fps:.0f} fps', disp.shape[0] - 25, (180, 180, 180), 0.7)
            cv2.imshow('Sign2Text live demo', disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                history, last = [], None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(ROOT, 'results', 'demo_model.pt'))
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='minimum confidence to show a phrase (else "?")')
    ap.add_argument('--camera', type=int, default=0)
    a = ap.parse_args()
    main(a.model, a.threshold, a.camera)
