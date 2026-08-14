from __future__ import annotations
import threading
import pyttsx3

class VoiceService:
    def __init__(self, rate: int = 165) -> None:
        self.rate = rate

    def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        def _run() -> None:
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        threading.Thread(target=_run, daemon=True).start()
