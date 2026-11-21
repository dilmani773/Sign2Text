"""
Sign2Text — Recognition engine (shared by the web UI)
=====================================================
Turns a stream of keypoint frames into conversation events. Same logic as
demo/live_demo.py: segment a sign when the hands go up and down, classify
it, and switch to fingerspelling after "my name is".

    engine = Engine(phrase_model, classes, speller)
    for kp, t in frames:
        engine.step(kp, t)          # kp: (543, 3) keypoints, t: seconds
        engine.snapshot(t)          # dict the UI shows

Models are passed in, so the engine is easy to test with fakes.
"""

import time
from typing import Callable, List, Optional

import numpy as np

SPELL_TRIGGER = 'my name is'
SPELL_PREFIX = 'My name is'
SPELL_TIMEOUT = 8.0          # leave spelling mode if nothing is spelled for this long
CAPTION_HOLD = 4.0           # seconds the newest caption stays big
QUESTIONS = {'how are you', 'what is your name', 'do you understand'}


def pretty(phrase: str) -> str:
    """'i am fine' → 'I am fine', 'how are you' → 'How are you?'"""
    words = ['I' if w == 'i' else w for w in phrase.strip().split()]
    text = ' '.join(words)
    text = text[:1].upper() + text[1:]
    text = text.replace(" i'", " I'")
    if phrase.lower().rstrip('?') in QUESTIONS and not text.endswith('?'):
        text += '?'
    return text


class Engine:
    """
    predict: function(list of segments) → probabilities (n, classes), any
             array-like. Default wiring uses s2t_demo.predict_proba.
    """

    def __init__(self, predict: Callable, classes: List[str], segmenter_factory: Callable,
                 speller=None, threshold: float = 0.5):
        self.predict, self.classes = predict, classes
        self.new_segmenter = segmenter_factory
        self.speller, self.threshold = speller, threshold
        self.seg = segmenter_factory()
        self.spelling, self.spell_prefix, self.spell_t = False, '', 0.0
        self.transcript: List[dict] = []
        self.caption: Optional[dict] = None
        self.last_guess: Optional[dict] = None          # latest classification, even if rejected
        self.version = 0                                # bumps when anything visible changes

    # ── commands from the UI ────────────────────────────────────────────────
    def clear(self) -> None:
        self.transcript, self.caption, self.last_guess = [], None, None
        self.version += 1

    def start_spelling(self, prefix: str = '', now: Optional[float] = None) -> bool:
        if not self.speller:
            return False
        self.speller.reset()
        self.spelling, self.spell_prefix = True, prefix
        self.spell_t = time.time() if now is None else now
        self.version += 1
        return True

    # ── per frame ───────────────────────────────────────────────────────────
    def step(self, kp: np.ndarray, now: float) -> None:
        if self.spelling:
            before = self.speller.spelled
            ev = self.speller.push(kp, now)
            if self.speller.spelled != before:
                self.version += 1
            if ev == 'END':
                self._finish_spelling(now, self.speller.spelled)
            elif not self.speller.spelled and now - self.spell_t > SPELL_TIMEOUT:
                self._finish_spelling(now, '')
            return

        was_signing = bool(self.seg.seg)
        segment = self.seg.push(kp, now)
        if bool(self.seg.seg) != was_signing:
            self.version += 1
        if segment is None:
            return

        prob = np.asarray(self.predict([segment]))[0]
        order = np.argsort(-prob)[:3]
        top3 = [(self.classes[int(j)], float(prob[j])) for j in order]
        name, p = top3[0]
        self.last_guess = {'phrase': name, 'conf': p,
                           'top3': [{'phrase': c, 'conf': v} for c, v in top3]}
        if name == '_idle':
            pass
        elif p >= self.threshold and name == SPELL_TRIGGER and self.speller:
            self.start_spelling(SPELL_PREFIX, now)
        elif p >= self.threshold:
            self._add(pretty(name), name, p, now, 'phrase')
        self.version += 1

    def _finish_spelling(self, now: float, word: str) -> None:
        self.spelling = False
        self.seg = self.new_segmenter()
        if word:
            text = f'{self.spell_prefix} {word.capitalize()}'.strip()
            self._add(text, SPELL_TRIGGER if self.spell_prefix else 'spelled', 1.0, now, 'spelled')
        self.version += 1

    def _add(self, text: str, phrase: str, conf: float, now: float, kind: str) -> None:
        item = {'text': text, 'phrase': phrase, 'conf': round(conf, 3),
                'time': time.strftime('%H:%M:%S', time.localtime(now if now > 1e8 else time.time())),
                'kind': kind, 'id': len(self.transcript) + 1 if not self.transcript
                else self.transcript[-1]['id'] + 1}
        self.transcript = (self.transcript + [item])[-30:]
        self.caption = {'text': text, 'conf': round(conf, 3), 'at': now, 'id': item['id']}

    # ── what the UI shows ───────────────────────────────────────────────────
    def snapshot(self, now: float) -> dict:
        if self.spelling:
            status = 'spelling'
        elif self.seg.seg:
            status = 'signing'
        else:
            status = 'ready'
        caption = self.caption if self.caption and now - self.caption['at'] < CAPTION_HOLD else None
        return {
            'version': self.version,
            'status': status,
            'caption': caption,
            'spelling': {'active': self.spelling, 'prefix': self.spell_prefix,
                         'letters': self.speller.spelled if self.spelling else ''},
            'transcript': self.transcript,
            'last_guess': self.last_guess,
            'can_spell': self.speller is not None,
            'phrases': [c for c in self.classes if c != '_idle'],
        }
