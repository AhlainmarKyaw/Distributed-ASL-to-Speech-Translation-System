from __future__ import annotations
import logging
import platform
import subprocess
import threading
import pyttsx3

logger = logging.getLogger(__name__)

class VoiceService:
    def __init__(self, rate: int = 165) -> None:
        self.rate = rate

    def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        def _run() -> None:
            try:
                if platform.system() == 'Darwin':
                    # The native command is reliable from a worker thread,
                    # unlike NSSpeechSynthesizer through pyttsx3 on macOS.
                    subprocess.run(
                        ['say', '-r', str(self.rate), text],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return
                engine = pyttsx3.init()
                engine.setProperty('rate', self.rate)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except (OSError, RuntimeError, subprocess.SubprocessError):
                logger.exception('Text-to-speech playback failed')
        threading.Thread(target=_run, daemon=True).start()
