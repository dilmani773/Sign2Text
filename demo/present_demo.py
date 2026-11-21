"""
Sign2Text — Fast presentation demo (Python window)
==================================================
The clean presentation look of live_demo.py --present, made faster:
the camera, MediaPipe and the screen run in separate threads, so the
video never waits for recognition and never shows old, queued frames.

Run from the repo root:
    python demo/present_demo.py
    python demo/present_demo.py --fast                 # slow laptop: lighter phrase recognition
    python demo/present_demo.py --debug                # print letter guesses while spelling
    python demo/present_demo.py --fullscreen
    python demo/present_demo.py --record videos/demo.mp4

Keys:  S = spell a name   C = clear   K = hand lines on/off   Q = quit
"""

import os
os.environ.setdefault('GLOG_minloglevel', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import sys
import time
import argparse
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'training'))
sys.path.insert(0, os.path.join(ROOT, 'demo'))


def main(model_path: str, letters_path: str, camera: int, threshold: float,
         fast: bool, fullscreen: bool, record_path: str = None, max_seconds: float = 0,
         debug: bool = False) -> None:
    import cv2
    import torch
    import mediapipe as mp
    from s2t_demo import PhraseClassifier, Segmenter, predict_proba
    from record_phrases import pad_to_16x9, results_to_543
    from engine import Engine
    from present import Presenter, VideoRecorder
    from web_demo import Grabber

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))      # leave cores for MediaPipe
    if not os.path.exists(model_path):
        sys.exit(f"Phrase model not found: {model_path}\n"
                 f"Train it with: python training/s2t_demo.py --data data data_friend")
    ck = torch.load(model_path, map_location='cpu', weights_only=False)
    model = PhraseClassifier(len(ck['classes']), ck['encoder_cfg'])
    model.load_state_dict(ck['model']); model.eval()
    speller = None
    if letters_path and os.path.exists(letters_path):
        from fingerspell import LetterModel, Speller
        speller = Speller(LetterModel.load(letters_path))

    def predict(segs):
        with torch.no_grad():
            return predict_proba(model, segs).numpy()

    engine = Engine(predict, ck['classes'], Segmenter, speller, threshold)
    print(f"Phrases: {len(ck['classes'])} | fingerspelling: {'on' if speller else 'off'} | "
          f"mode: {'fast' if fast else 'normal'}")

    cap = cv2.VideoCapture(camera)
    # Always full size from the camera: letters (like U vs V) need a sharp hand.
    # --fast only makes the phrase part lighter (see recognise()).
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {camera}. Try --camera 1")
    stop = threading.Event()
    grab = Grabber(cap, stop); grab.start()

    mp_h, mp_draw = mp.solutions.holistic, mp.solutions.drawing_utils
    line = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
    dots = mp_draw.DrawingSpec(color=(80, 200, 255), thickness=1, circle_radius=2)
    arm_pairs = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]   # shoulders, elbows, wrists
    lock = threading.Lock()

    def draw_arms(img, pose):
        if pose is None:
            return
        H, W = img.shape[:2]
        lm = pose.landmark
        for a, b in arm_pairs:
            if lm[a].visibility > 0.4 and lm[b].visibility > 0.4:
                pa = (int(lm[a].x * W), int(lm[a].y * H)); pb = (int(lm[b].x * W), int(lm[b].y * H))
                cv2.line(img, pa, pb, (255, 255, 255), 2, cv2.LINE_AA)
        for i in (11, 12, 13, 14, 15, 16):
            if lm[i].visibility > 0.4:
                cv2.circle(img, (int(lm[i].x * W), int(lm[i].y * H)), 4, (80, 200, 255), -1, cv2.LINE_AA)
    shared = {'res': None, 'snap': engine.snapshot(time.time()), 'fps': 0.0, 'cmds': []}

    # ── recognition thread: newest frame → MediaPipe → engine ───────────────
    # Fingerspelling always uses the full camera image and the full MediaPipe
    # model, the same as fingerspell.py live (the letter model was trained
    # that way). --fast only lightens phrase recognition.
    def recognise():
        last_n, fps, fps_t = -1, 0.0, time.time()
        opts = dict(refine_face_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        hol_full = mp_h.Holistic(model_complexity=1, **opts)
        hol_lite = mp_h.Holistic(model_complexity=0, **opts) if fast else None
        try:
            while not stop.is_set():
                frame, t_cap, n = grab.latest()
                if frame is None or n == last_n:
                    time.sleep(0.003); continue
                last_n = n
                if hol_lite is not None and not engine.spelling:
                    small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
                    res = hol_lite.process(cv2.cvtColor(pad_to_16x9(small), cv2.COLOR_BGR2RGB))
                else:
                    res = hol_full.process(cv2.cvtColor(pad_to_16x9(frame), cv2.COLOR_BGR2RGB))
                now = time.time()
                fps = 0.9 * fps + 0.1 / max(now - fps_t, 1e-3); fps_t = now
                with lock:
                    cmds, shared['cmds'] = shared['cmds'], []
                for c in cmds:
                    if c == 'clear':
                        engine.clear()
                    elif c == 'spell' and not engine.spelling:
                        engine.start_spelling('', now)
                engine.step(results_to_543(res), t_cap)
                with lock:
                    shared['res'], shared['snap'], shared['fps'] = res, engine.snapshot(now), fps
                    shared['guess'] = speller.current if (speller and engine.spelling) else None
        finally:
            hol_full.close()
            if hol_lite is not None:
                hol_lite.close()

    threading.Thread(target=recognise, daemon=True).start()

    # ── screen (main thread): newest frame + latest hand lines + captions ──
    presenter, recorder = Presenter(), None
    presenter.show_transcript = False              # only the current phrase on screen
    win = 'Sign2Text'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(win, 1280, 720)
    seen_ids, was_spelling, last_n, t_start, last_print = set(), False, -1, time.time(), 0.0
    last_guess = None

    try:
        while True:
            if grab.failed:
                print("The webcam stopped sending frames."); break
            frame, _, n = grab.latest()
            if frame is None or n == last_n:
                key = cv2.waitKey(3) & 0xFF
            else:
                last_n = n
                now = time.time()
                with lock:
                    res, snap, fps = shared['res'], shared['snap'], shared['fps']

                # feed the presenter from the engine's state
                for item in snap['transcript']:
                    if item['id'] not in seen_ids:
                        seen_ids.add(item['id'])
                        presenter.finish_live(item['text'], now)
                sp = snap['spelling']
                if sp['active']:
                    presenter.show_live(f"{sp['prefix']} {sp['letters']}".strip() or '...', now)
                elif was_spelling and presenter.live_text is not None:
                    presenter.cancel_live()
                was_spelling = sp['active']

                raw_h, raw_w = frame.shape[:2]
                disp = pad_to_16x9(frame) if (presenter.show_skeleton and res is not None) else frame
                if presenter.show_skeleton and res is not None:
                    disp = disp.copy()
                    draw_arms(disp, res.pose_landmarks)
                    for hand in (res.left_hand_landmarks, res.right_hand_landmarks):
                        if hand is not None:
                            mp_draw.draw_landmarks(disp, hand, mp_h.HAND_CONNECTIONS, dots, line)
                    ph, pw = disp.shape[:2]
                    x0, y0 = (pw - raw_w) // 2, (ph - raw_h) // 2
                    disp = disp[y0:y0 + raw_h, x0:x0 + raw_w]
                disp = cv2.flip(disp, 1)
                if disp.shape[0] != 720:                                   # same look at any camera size
                    disp = cv2.resize(disp, (int(720 * disp.shape[1] / disp.shape[0]), 720))
                disp = presenter.render(disp, now, signing=snap['status'] in ('signing', 'spelling'))
                if record_path:
                    if recorder is None:
                        recorder = VideoRecorder(record_path, (disp.shape[1], disp.shape[0]))
                    recorder.write(disp, now)
                cv2.imshow(win, disp)
                if now - last_print > 5:
                    print(f"recognition speed: {fps:.1f} frames per second")
                    last_print = now
                if debug:
                    g = shared.get('guess')
                    if g and g != last_guess:
                        print(f"  letter guess: {g[0]}  {g[1]:.0%}   (finger gap {g[2]:.2f})")
                        last_guess = g
                key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            if key == ord('c'):
                presenter.clear()
                with lock: shared['cmds'].append('clear')
            if key == ord('s') and speller:
                with lock: shared['cmds'].append('spell')
            if key == ord('k'):
                presenter.show_skeleton = not presenter.show_skeleton
            if max_seconds and time.time() - t_start > max_seconds:
                break
    finally:
        stop.set()
        if recorder:
            recorder.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(ROOT, 'results', 'demo_model.pt'))
    ap.add_argument('--letters', default=os.path.join(ROOT, 'results', 'letters_model.pt'))
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--fast', action='store_true', help='slow laptop: 640x360 camera, light MediaPipe')
    ap.add_argument('--fullscreen', action='store_true')
    ap.add_argument('--record', default=None, metavar='FILE.mp4')
    ap.add_argument('--debug', action='store_true', help='print each letter guess while spelling')
    a = ap.parse_args()
    main(a.model, a.letters, a.camera, a.threshold, a.fast, a.fullscreen, a.record, debug=a.debug)
