"""
Sign2Text — Web demo (runs on your laptop, opens in the browser)
================================================================
Same recognition as demo/live_demo.py, shown in a browser page:
live camera, captions, the conversation so far, and the phrases it knows.

Setup (once):
    pip install fastapi "uvicorn[standard]"

Run from the repo root:
    python demo/web_demo.py                     # opens http://127.0.0.1:8000
    python demo/web_demo.py --camera 1          # another webcam
    python demo/web_demo.py --threshold 0.4     # accept less confident phrases
    python demo/web_demo.py --fast              # slow laptop: lighter camera and MediaPipe
    python demo/web_demo.py --fake              # try the page without camera or models

In the page:
    S = spell a name    C = clear    H = hand lines on/off    V = read aloud on/off
"""

import os
os.environ.setdefault('GLOG_minloglevel', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import sys
import json
import time
import queue
import asyncio
import argparse
import threading
import webbrowser
from typing import Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'training'))
sys.path.insert(0, os.path.join(ROOT, 'demo'))
PAGE = os.path.join(ROOT, 'demo', 'web', 'index.html')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — SHARED STATE between the camera thread and the web server
# ─────────────────────────────────────────────────────────────────────────────

class Hub:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg: Optional[bytes] = None
        self.frame_no = 0
        self.state: dict = {'status': 'starting', 'transcript': [], 'phrases': []}
        self.commands: 'queue.Queue[str]' = queue.Queue()
        self.error: Optional[str] = None

    def put_frame(self, jpeg: bytes) -> None:
        with self.lock:
            self.jpeg, self.frame_no = jpeg, self.frame_no + 1

    def put_state(self, state: dict) -> None:
        with self.lock:
            self.state = state

    def get(self):
        with self.lock:
            return self.jpeg, self.frame_no, dict(self.state)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CAMERA LOOP (real models)
# ─────────────────────────────────────────────────────────────────────────────

class Grabber(threading.Thread):
    """
    Reads the webcam as fast as it can and keeps only the newest frame,
    so the processing loop never works on an old, queued frame.
    """

    def __init__(self, cap, stop: threading.Event):
        super().__init__(daemon=True)
        self.cap, self.stop = cap, stop
        self.lock = threading.Lock()
        self.frame, self.t, self.n, self.failed = None, 0.0, 0, False

    def run(self):
        while not self.stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self.failed = True
                return
            with self.lock:
                self.frame, self.t, self.n = frame, time.time(), self.n + 1

    def latest(self):
        with self.lock:
            return self.frame, self.t, self.n


def camera_loop(hub: Hub, model_path: str, letters_path: str, camera: int,
                threshold: float, stop: threading.Event, fast: bool = False) -> None:
    """
    Three parts run side by side, so nothing waits for the slowest step:
        Grabber thread   webcam → newest frame
        video thread     newest frame + latest hand lines → JPEG for the page (camera speed)
        this loop        newest frame → MediaPipe → recognition (as fast as the CPU allows)
    """
    try:
        import cv2
        import torch
        import mediapipe as mp
        from s2t_demo import PhraseClassifier, Segmenter, predict_proba
        from record_phrases import pad_to_16x9, results_to_543
        from engine import Engine

        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))     # leave cores for MediaPipe
        if not os.path.exists(model_path):
            raise RuntimeError(f"Phrase model not found: {model_path}. "
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
        print(f"Phrases: {model_path} ({len(ck['classes'])} classes) | "
              f"fingerspelling: {'on' if speller else 'off'} | mode: {'fast' if fast else 'normal'}")

        cap = cv2.VideoCapture(camera)
        w, h = (640, 360) if fast else (1280, 720)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera}. Try --camera 1, "
                               f"and close other apps that use the webcam.")
        grab = Grabber(cap, stop); grab.start()

        mp_h, mp_draw = mp.solutions.holistic, mp.solutions.drawing_utils
        line = mp_draw.DrawingSpec(color=(238, 242, 239), thickness=2, circle_radius=1)
        dots = mp_draw.DrawingSpec(color=(66, 185, 244), thickness=2, circle_radius=3)
        shared = {'res': None, 'show_hands': True}

        def video_thread():
            last_n = -1
            while not stop.is_set():
                frame, _, n = grab.latest()
                if frame is None or n == last_n:
                    time.sleep(0.005); continue
                last_n = n
                raw_h, raw_w = frame.shape[:2]
                padded = pad_to_16x9(frame).copy()   # never draw on the frame MediaPipe reads
                res = shared['res']
                if shared['show_hands'] and res is not None:
                    for hand in (res.left_hand_landmarks, res.right_hand_landmarks):
                        if hand is not None:
                            mp_draw.draw_landmarks(padded, hand, mp_h.HAND_CONNECTIONS, dots, line)
                ph, pw = padded.shape[:2]
                x0, y0 = (pw - raw_w) // 2, (ph - raw_h) // 2
                disp = cv2.flip(padded[y0:y0 + raw_h, x0:x0 + raw_w], 1)
                if disp.shape[1] > 960:                                   # lighter stream
                    disp = cv2.resize(disp, (960, int(960 * disp.shape[0] / disp.shape[1])))
                ok, buf = cv2.imencode('.jpg', disp, [cv2.IMWRITE_JPEG_QUALITY, 78])
                if ok:
                    hub.put_frame(buf.tobytes())

        threading.Thread(target=video_thread, daemon=True).start()

        fps, fps_t, last_n = 0.0, time.time(), -1
        with mp_h.Holistic(model_complexity=0 if fast else 1, refine_face_landmarks=False,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5) as hol:
            while not stop.is_set():
                if grab.failed:
                    raise RuntimeError("The webcam stopped sending frames.")
                frame, t_cap, n = grab.latest()
                if frame is None or n == last_n:
                    time.sleep(0.003); continue
                last_n = n
                res = hol.process(cv2.cvtColor(pad_to_16x9(frame), cv2.COLOR_BGR2RGB))
                shared['res'] = res
                now = time.time()
                fps = 0.9 * fps + 0.1 / max(now - fps_t, 1e-3); fps_t = now

                while not hub.commands.empty():
                    cmd = hub.commands.get_nowait()
                    if cmd == 'clear':
                        engine.clear()
                    elif cmd == 'spell' and not engine.spelling:
                        engine.start_spelling('', now)
                    elif cmd == 'hands':
                        shared['show_hands'] = not shared['show_hands']

                engine.step(results_to_543(res), t_cap)             # capture time keeps timing exact
                snap = engine.snapshot(now)
                snap.update({'fps': round(fps, 1), 'show_hands': shared['show_hands'], 'error': None})
                hub.put_state(snap)
        cap.release()
    except Exception as e:                                   # shown in the page
        hub.error = str(e)
        hub.put_state({'status': 'error', 'error': str(e), 'transcript': [], 'phrases': []})
        print("ERROR:", e)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FAKE LOOP (try the page without camera or models)
# ─────────────────────────────────────────────────────────────────────────────

FAKE_PHRASES = ['hello', 'thank you', 'please', 'sorry', 'yes', 'no', 'help me', 'how are you',
                'i am fine', 'what is your name', 'nice to meet you', 'good morning', 'goodbye',
                "i don't understand", 'again please', 'I am good', 'i understand', 'bad',
                'i love you', 'do you understand?', 'my name is']
FAKE_SCRIPT = [('hello', 0.97), ('how are you', 0.91), ('i am fine', 0.88), ('SPELL', 'HIYUMI'),
               ('nice to meet you', 0.93), ('goodbye', 0.95)]


def fake_frame(t: float) -> bytes:
    import cv2
    h, w = 720, 1280
    y = np.linspace(0, 1, h)[:, None]
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = (60 + 40 * y).astype(np.uint8)
    img[..., 1] = (70 + 30 * y).astype(np.uint8)
    img[..., 2] = (74 + 20 * y).astype(np.uint8)
    cv2.ellipse(img, (640, 300), (120, 150), 0, 0, 360, (150, 160, 175), -1)
    cv2.rectangle(img, (420, 470), (860, 720), (110, 90, 80), -1)
    cx = int(820 + 60 * np.sin(t * 2))
    cv2.circle(img, (cx, 430), 38, (170, 185, 200), -1)
    return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()


def fake_loop(hub: Hub, stop: threading.Event) -> None:
    from engine import pretty, CAPTION_HOLD
    transcript, caption, t0, step = [], None, time.time(), 0
    spelling, letters, status = False, '', 'ready'
    next_at = t0 + 2.0
    while not stop.is_set():
        now = time.time()
        while not hub.commands.empty():
            if hub.commands.get_nowait() == 'clear':
                transcript, caption = [], None
        if now >= next_at:
            item = FAKE_SCRIPT[step % len(FAKE_SCRIPT)]
            if item[0] == 'SPELL':
                if not spelling:
                    spelling, letters = True, ''
                if len(letters) < len(item[1]):
                    letters = item[1][:len(letters) + 1]; next_at = now + 0.8
                else:
                    text = f'My name is {letters.capitalize()}'
                    transcript.append({'id': len(transcript) + 1, 'text': text, 'phrase': 'my name is',
                                       'conf': 1.0, 'time': time.strftime('%H:%M:%S'), 'kind': 'spelled'})
                    caption = {'text': text, 'conf': 1.0, 'at': now, 'id': len(transcript)}
                    spelling, step, next_at = False, step + 1, now + 4.0
            else:
                text = pretty(item[0])
                transcript.append({'id': len(transcript) + 1, 'text': text, 'phrase': item[0],
                                   'conf': item[1], 'time': time.strftime('%H:%M:%S'), 'kind': 'phrase'})
                caption = {'text': text, 'conf': item[1], 'at': now, 'id': len(transcript)}
                step, next_at = step + 1, now + 4.0
        status = 'spelling' if spelling else ('signing' if 0 < next_at - now < 1.4 else 'ready')
        hub.put_frame(fake_frame(now - t0))
        hub.put_state({
            'version': len(transcript) * 100 + len(letters), 'status': status,
            'caption': caption if caption and now - caption['at'] < CAPTION_HOLD else None,
            'spelling': {'active': spelling, 'prefix': 'My name is', 'letters': letters},
            'transcript': transcript[-30:],
            'last_guess': {'phrase': transcript[-1]['phrase'], 'conf': transcript[-1]['conf'],
                           'top3': []} if transcript else None,
            'can_spell': True, 'phrases': FAKE_PHRASES, 'fps': 15.0, 'show_hands': True,
            'error': None, 'fake': True})
        time.sleep(1 / 15)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — WEB SERVER
# ─────────────────────────────────────────────────────────────────────────────

def make_app(hub: Hub):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, StreamingResponse

    app = FastAPI(title='Sign2Text')

    @app.get('/')
    def index():
        return FileResponse(PAGE, media_type='text/html')

    @app.get('/video')
    async def video():
        async def gen():
            last = -1
            while True:
                jpeg, n, _ = hub.get()
                if jpeg is not None and n != last:
                    last = n
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                           str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n')
                await asyncio.sleep(0.02)
        return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')

    @app.get('/state')
    def state():
        return hub.get()[2]

    @app.websocket('/ws')
    async def ws(sock: WebSocket):
        await sock.accept()

        async def receive():
            try:
                while True:
                    msg = await sock.receive_text()
                    cmd = json.loads(msg).get('cmd') if msg.startswith('{') else msg
                    if cmd in ('clear', 'spell', 'hands'):
                        hub.commands.put(cmd)
            except (WebSocketDisconnect, RuntimeError):
                pass

        task = asyncio.create_task(receive())
        last = None
        try:
            while not task.done():
                s = hub.get()[2]
                payload = json.dumps(s, default=float)
                if payload != last:
                    await sock.send_text(payload)
                    last = payload
                await asyncio.sleep(0.1)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            task.cancel()

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(ROOT, 'results', 'demo_model.pt'))
    ap.add_argument('--letters', default=os.path.join(ROOT, 'results', 'letters_model.pt'))
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--fake', action='store_true', help='no camera or models, scripted demo')
    ap.add_argument('--fast', action='store_true',
                    help='for slow laptops: 640x360 camera and the light MediaPipe model')
    ap.add_argument('--no-browser', action='store_true')
    a = ap.parse_args()

    import uvicorn
    hub, stop = Hub(), threading.Event()
    target = (lambda: fake_loop(hub, stop)) if a.fake else \
        (lambda: camera_loop(hub, a.model, a.letters, a.camera, a.threshold, stop, a.fast))
    threading.Thread(target=target, daemon=True).start()
    url = f'http://127.0.0.1:{a.port}'
    print(f"Sign2Text web demo: {url}   (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(make_app(hub), host='127.0.0.1', port=a.port, log_level='warning')
    finally:
        stop.set()


if __name__ == '__main__':
    main()
