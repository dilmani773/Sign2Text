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

Presentation mode for demo videos (clean subtitles, no technical text):
    python demo/live_demo.py --present
    python demo/live_demo.py --present --record videos/demo.mp4 --fullscreen
    Keys in presentation mode:  K = skeleton on/off   C = clear   Q = quit

Debug (shows the top 3 guesses, saves every segment for later checking):
    python demo/live_demo.py --debug --save live_segments

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


def main(model_path: str, threshold: float, camera: int, debug: bool = False,
         save_dir: str = None, present: bool = False, record_path: str = None,
         fullscreen: bool = False) -> None:
    import cv2
    import torch
    import mediapipe as mp
    from s2t_demo import PhraseClassifier, Segmenter, predict_proba
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

    presenter = recorder = None
    if present or record_path:
        from present import Presenter, VideoRecorder
        presenter = Presenter()
    win = 'Sign2Text live demo'
    if fullscreen:
        cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    seg = Segmenter()
    history, last, last_until, top3 = [], None, 0.0, []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fps, fps_t = 0.0, time.time()

    with mp_h.Holistic(model_complexity=1, refine_face_landmarks=False,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw_h, raw_w = frame.shape[:2]
            frame = pad_to_16x9(frame)
            res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - fps_t, 1e-3); fps_t = now
            kp = results_to_543(res)

            segment = seg.push(kp, now)
            if segment is not None:
                prob = predict_proba(model, [segment])[0]          # works for either dominant hand
                p, i = prob.max(0)
                name = classes[int(i)]
                vals, idx = prob.topk(3)
                top3 = [(classes[int(j)], float(v)) for v, j in zip(vals, idx)]
                if debug:
                    print(f"{len(segment):3d} frames  " +
                          "  |  ".join(f"{c} {100 * v:.0f}%" for c, v in top3))
                if save_dir:
                    fn = time.strftime('%Y%m%d_%H%M%S') + f'_{name.replace(" ", "_").replace("?", "")}.npz'
                    np.savez_compressed(os.path.join(save_dir, fn), kp=segment.astype(np.float32),
                                        top3=np.array([c for c, _ in top3]))
                if name == '_idle':
                    last = None
                elif p.item() >= threshold:
                    last = (name, p.item())
                    history = (history + [name])[-6:]
                    if presenter:
                        presenter.show_phrase(name, now)
                else:
                    last = ('?', p.item())
                last_until = now + 3.0

            if presenter:
                disp = frame.copy()
                if presenter.show_skeleton:
                    thin = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
                    dots = mp_draw.DrawingSpec(color=(80, 200, 255), thickness=1, circle_radius=2)
                    for hand in (res.left_hand_landmarks, res.right_hand_landmarks):
                        mp_draw.draw_landmarks(disp, hand, mp_h.HAND_CONNECTIONS, dots, thin)
                ph, pw = disp.shape[:2]                        # remove 16:9 padding bars
                x0, y0 = (pw - raw_w) // 2, (ph - raw_h) // 2
                disp = cv2.flip(disp[y0:y0 + raw_h, x0:x0 + raw_w], 1)
                disp = presenter.render(disp, now, signing=bool(seg.seg))
                if record_path:
                    if recorder is None:
                        recorder = VideoRecorder(record_path, (disp.shape[1], disp.shape[0]))
                    recorder.write(disp, now)
                cv2.imshow(win, disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('c'):
                    presenter.clear(); history, last = [], None
                if key == ord('k'):
                    presenter.show_skeleton = not presenter.show_skeleton
                continue

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
            if debug and top3 and now < last_until:
                for k, (c, v) in enumerate(top3):
                    put(disp, f'{k + 1}. {c}  {100 * v:.0f}%', 160 + 32 * k, (200, 255, 200), 0.7, 2)
            if history:
                put(disp, ' | '.join(history), disp.shape[0] - 70, (255, 255, 255), 0.8)
            put(disp, f'C = clear   Q = quit   |   {fps:.0f} fps', disp.shape[0] - 25, (180, 180, 180), 0.7)
            cv2.imshow(win, disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                history, last = [], None

    if recorder:
        recorder.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(ROOT, 'results', 'demo_model.pt'))
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='minimum confidence to show a phrase (else "?")')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--debug', action='store_true', help='show and print the top 3 guesses')
    ap.add_argument('--save', default=None, metavar='DIR', help='save every live segment here')
    ap.add_argument('--present', action='store_true', help='clean presentation look for demo videos')
    ap.add_argument('--record', default=None, metavar='FILE.mp4', help='save the presentation view as a video')
    ap.add_argument('--fullscreen', action='store_true')
    a = ap.parse_args()
    main(a.model, a.threshold, a.camera, a.debug, a.save, a.present, a.record, a.fullscreen)