"""Text-to-speech tests with both the engine and thread boundary mocked."""

from __future__ import annotations

from unittest.mock import Mock

import client.voice_service as voice_module
from client.voice_service import VoiceService


class ImmediateThread:
    def __init__(self, *, target, daemon):
        self.target = target
        self.daemon = daemon

    def start(self):
        self.target()


def test_speak_uses_mock_engine_without_audio_device(monkeypatch) -> None:
    engine = Mock()
    monkeypatch.setattr(voice_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(voice_module.pyttsx3, "init", lambda: engine)
    monkeypatch.setattr(voice_module.threading, "Thread", ImmediateThread)

    VoiceService(rate=180).speak("  Hello  ")

    engine.setProperty.assert_called_once_with("rate", 180)
    engine.say.assert_called_once_with("Hello")
    engine.runAndWait.assert_called_once_with()
    engine.stop.assert_called_once_with()


def test_blank_text_does_not_initialize_speech_engine(monkeypatch) -> None:
    initialize = Mock()
    monkeypatch.setattr(voice_module.pyttsx3, "init", initialize)

    VoiceService().speak("   ")

    initialize.assert_not_called()


def test_macos_speak_uses_native_say_without_shell(monkeypatch) -> None:
    run = Mock()
    monkeypatch.setattr(voice_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(voice_module.subprocess, "run", run)
    monkeypatch.setattr(voice_module.threading, "Thread", ImmediateThread)

    VoiceService(rate=175).speak("  Hello world  ")

    run.assert_called_once_with(
        ["say", "-r", "175", "Hello world"],
        check=True,
        stdout=voice_module.subprocess.DEVNULL,
        stderr=voice_module.subprocess.DEVNULL,
    )
