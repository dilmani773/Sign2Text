"""
Sign2Text — Presentation overlay for demo videos
================================================
Clean look for recording a demo (LinkedIn, presentations):
    - phrase shown as a large subtitle at the bottom, fades out after a few seconds
    - short transcript of the last phrases, like a conversation
    - small title and a "signing" dot, nothing technical on screen
    - optional thin skeleton overlay (toggle with K)

Used by:  python demo/live_demo.py --present [--record demo.mp4] [--fullscreen]
"""

import os
from typing import List, Optional, Tuple

import numpy as np

QUESTIONS = {'how are you', 'what is your name', 'do you understand?'}

FONT_CANDIDATES = [
    'C:/Windows/Fonts/segoeuib.ttf', 'C:/Windows/Fonts/segoeui.ttf',      # Windows
    'C:/Windows/Fonts/arialbd.ttf', 'C:/Windows/Fonts/arial.ttf',
    '/System/Library/Fonts/Supplemental/Arial Bold.ttf',                  # Mac
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',               # Linux
]
FONT_REGULAR = [
    'C:/Windows/Fonts/segoeui.ttf', 'C:/Windows/Fonts/arial.ttf',
    '/System/Library/Fonts/Supplemental/Arial.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]


def pretty(phrase: str) -> str:
    """'i am fine' → 'I am fine', 'how are you' → 'How are you?'"""
    words = ['I' if w == 'i' else w for w in phrase.strip().split()]
    text = ' '.join(words)
    text = text[:1].upper() + text[1:]
    text = text.replace(" i'", " I'").replace("I don't", "I don't")
    if phrase.lower().rstrip('?') in {q.rstrip('?') for q in QUESTIONS} and not text.endswith('?'):
        text += '?'
    return text


def _font(paths: List[str], size: int):
    from PIL import ImageFont
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


class Presenter:
    """Draws the presentation overlay. Call show_phrase() when a phrase is recognised."""

    def __init__(self, title: str = 'Sign2Text  |  ASL to English', hold_s: float = 3.5,
                 fade_s: float = 0.6, transcript_len: int = 4):
        self.title = title
        self.hold_s, self.fade_s = hold_s, fade_s
        self.transcript_len = transcript_len
        self.current: Optional[Tuple[str, float]] = None      # (text, time shown)
        self.transcript: List[str] = []
        self.show_skeleton = True
        self.show_transcript = True          # small line of earlier phrases above the caption
        self.live_text: Optional[str] = None                   # text being spelled right now
        self._fonts = {}

    def font(self, kind: str, size: int):
        key = (kind, size)
        if key not in self._fonts:
            self._fonts[key] = _font(FONT_CANDIDATES if kind == 'bold' else FONT_REGULAR, size)
        return self._fonts[key]

    def show_phrase(self, phrase: str, now: float) -> None:
        text = pretty(phrase)
        self.current = (text, now)
        self.transcript = (self.transcript + [text])[-self.transcript_len:]

    def show_live(self, text: str, now: float) -> None:
        """Text that updates while a name is being spelled ('My name is HIY')."""
        self.live_text = text

    def finish_live(self, text: str, now: float) -> None:
        self.live_text = None
        self.current = (text, now)
        self.transcript = (self.transcript + [text])[-self.transcript_len:]

    def cancel_live(self) -> None:
        self.live_text = None

    def clear(self) -> None:
        self.current, self.transcript, self.live_text = None, [], None

    def _alpha(self, now: float) -> float:
        if not self.current:
            return 0.0
        age = now - self.current[1]
        if age < 0.15:
            return age / 0.15                                  # quick fade in
        if age < self.hold_s:
            return 1.0
        return max(0.0, 1.0 - (age - self.hold_s) / self.fade_s)

    def render(self, frame_bgr: np.ndarray, now: float, signing: bool) -> np.ndarray:
        from PIL import Image, ImageDraw
        h, w = frame_bgr.shape[:2]
        s = h / 720.0                                          # scale for any resolution
        img = Image.fromarray(frame_bgr[:, :, ::-1]).convert('RGBA')
        layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)

        # top: title pill + signing indicator
        ft = self.font('bold', int(24 * s))
        tb = d.textbbox((0, 0), self.title, font=ft)
        pad = int(14 * s)
        d.rounded_rectangle((int(24 * s), int(22 * s), int(24 * s) + tb[2] + 2 * pad,
                             int(22 * s) + tb[3] + 2 * pad), radius=int(18 * s), fill=(15, 15, 20, 150))
        d.text((int(24 * s) + pad, int(22 * s) + pad - tb[1] // 2), self.title, font=ft,
               fill=(255, 255, 255, 235))
        if signing:
            r = int(9 * s); cx, cy = w - int(150 * s), int(46 * s)
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(235, 64, 52, 230))
            d.text((cx + 2 * r, cy - int(15 * s)), 'signing', font=self.font('regular', int(22 * s)),
                   fill=(255, 255, 255, 220))

        # bottom: transcript (small) + current phrase (large subtitle)
        a = self._alpha(now)
        live = self.live_text is not None
        if live:
            a = 1.0
        band_h = int(150 * s)
        if a > 0 or (self.show_transcript and self.transcript):
            d.rectangle((0, h - band_h, w, h), fill=(10, 10, 15, int(150 * max(a, 0.55))))
        cur_in_list = a > 0 and not live          # the newest phrase is shown big, not in the list
        if self.show_transcript and len(self.transcript) > (1 if cur_in_list else 0):
            past = self.transcript[:-1] if cur_in_list else self.transcript
            line = '   ·   '.join(past)
            fs = self.font('regular', int(22 * s))
            d.text((int(36 * s), h - band_h + int(14 * s)), line, font=fs, fill=(210, 210, 215, 200))
        if a > 0:
            text = self.live_text + ('_' if int(now * 2) % 2 == 0 else ' ') if live else self.current[0]
            size = int(58 * s)
            fb = self.font('bold', size)
            while d.textlength(text, font=fb) > w - 80 * s and size > 20:
                size -= 4; fb = self.font('bold', size)
            tw = d.textlength(text, font=fb)
            y = h - int(92 * s)
            d.text(((w - tw) / 2 + 2, y + 2), text, font=fb, fill=(0, 0, 0, int(140 * a)))
            d.text(((w - tw) / 2, y), text, font=fb, fill=(255, 255, 255, int(255 * a)))

        out = Image.alpha_composite(img, layer).convert('RGB')
        return np.asarray(out)[:, :, ::-1].copy()


class VideoRecorder:
    """Writes presentation frames to an .mp4 at a steady frame rate."""

    def __init__(self, path: str, size: Tuple[int, int], fps: float = 15.0):
        import cv2
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.fps = fps
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        self.t0, self.n = None, 0
        self.path = path

    def write(self, frame: np.ndarray, now: float) -> None:
        """Duplicates or drops frames so the video plays at real speed."""
        if self.t0 is None:
            self.t0 = now
        due = int((now - self.t0) * self.fps) + 1
        while self.n < due:
            self.writer.write(frame)
            self.n += 1

    def close(self) -> None:
        self.writer.release()
        print(f"Saved video: {self.path} ({self.n / self.fps:.1f} s)")
