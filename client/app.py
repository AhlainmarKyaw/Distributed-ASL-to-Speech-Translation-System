from __future__ import annotations

import logging
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk

from client.alphabet_buffer import AlphabetBuffer
from client.phrase_buffer import PhraseBuffer
from client.api_client import (
    post_alphabet,
    post_phrase_sequence,
    post_phrase_text,
)
from client.camera import Camera
from client.voice_service import VoiceService
from client.service_status import ServiceStatus, StatusPoller
from config import settings
from observability import get_logger, log_event, new_request_id


API = settings.api_url
logger = get_logger("client")

COMMON_PHRASES = [
    "Hello",
    "Thank you",
    "Good morning",
    "Please",
    "Yes",
    "No",
    "Help me",
    "Sorry",
    "I love you",
    "How are you?",
    "Nice to meet you",
    "Goodbye",
]


class FlatButton(tk.Label):
    """A consistently coloured button on macOS, Windows, and Linux."""

    def __init__(
        self,
        parent,
        *,
        text,
        command,
        color,
        compact=False,
    ):
        self.command = command
        self.normal_color = color

        super().__init__(
            parent,
            text=text,
            bg=color,
            fg="white",
            cursor="hand2",
            font=(
                "Segoe UI",
                9 if compact else 10,
                "bold",
            ),
            padx=8 if compact else 14,
            pady=4 if compact else 9,
            takefocus=True,
        )

        self.bind("<Button-1>", self._invoke)
        self.bind("<Return>", self._invoke)
        self.bind("<space>", self._invoke)

        self.bind(
            "<Enter>",
            lambda _event: self.config(
                bg=self._lighter(
                    self.normal_color
                )
            ),
        )

        self.bind(
            "<Leave>",
            lambda _event: self.config(
                bg=self.normal_color
            ),
        )

    def _invoke(self, _event=None):
        self.command()

    @staticmethod
    def _lighter(color):
        red, green, blue = (
            int(
                color[index:index + 2],
                16,
            )
            for index in (
                1,
                3,
                5,
            )
        )

        return (
            f"#{min(red + 24, 255):02x}"
            f"{min(green + 24, 255):02x}"
            f"{min(blue + 24, 255):02x}"
        )


class ASLApp:
    def __init__(
        self,
        root: tk.Tk,
    ) -> None:

        self.root = root

        self.root.title(
            "Distributed ASL-to-Speech Translation System"
        )

        self.root.geometry(
            "1280x780"
        )

        self.root.minsize(
            1080,
            700,
        )

        self.root.configure(
            bg="#08111f"
        )

        # =====================================================
        # CAMERA + BUFFERS
        # =====================================================

        self.camera = Camera(0)

        # Existing alphabet/fingerspelling buffer
        self.buffer = AlphabetBuffer()

        # New dynamic phrase temporal buffer
        self.phrase_buffer = PhraseBuffer()

        self.voice = VoiceService()

        # =====================================================
        # RECOGNITION MODE
        # =====================================================

        # "alphabet" or "phrase"
        self.recognition_mode = "alphabet"

        # =====================================================
        # CAMERA STATE
        # =====================================================

        self.running = False

        # Alphabet request control
        self.alphabet_inflight = False
        self.last_alphabet_submit = 0.0

        # Phrase request control
        self.phrase_inflight = False
        self.last_phrase_submit = 0.0

        # Prevent repeated identical phrase output
        self.last_phrase = None
        self.last_phrase_emit = 0.0

        # =====================================================
        # ALPHABET STABILIZATION
        # =====================================================

        self.last_letter = None
        self.streak = 0
        self.last_emit = 0.0

        # =====================================================
        # RESULT QUEUES
        # =====================================================

        self.result_queue: queue.Queue = (
            queue.Queue()
        )

        self.status_queue: queue.Queue = (
            queue.Queue(
                maxsize=1
            )
        )

        # =====================================================
        # DISTRIBUTED TASK INFORMATION
        # =====================================================

        self.last_request_id = None
        self.last_worker_id = None
        self.last_processing_ms = None
        self.last_task_status = None

        self.closing = False

        self._build_ui()

        # =====================================================
        # SERVICE STATUS POLLER
        # =====================================================

        self.status_poller = StatusPoller(
            self.status_queue,
            api_url=API,
        )

        self.status_poller.start()

        self.root.after(
            200,
            self._drain_service_status,
        )

    # =========================================================
    # UI
    # =========================================================

    def _build_ui(self) -> None:

        top = tk.Frame(
            self.root,
            bg="#08111f",
            padx=28,
            pady=22,
        )

        top.pack(
            fill="x"
        )

        tk.Label(
            top,
            text="Distributed ASL-to-Speech",
            bg="#08111f",
            fg="#6ee7ff",
            font=(
                "Segoe UI",
                27,
                "bold",
            ),
        ).pack(
            side="left"
        )

        tk.Label(
            top,
            text="Translation System",
            bg="#08111f",
            fg="#d7e4f5",
            font=(
                "Segoe UI",
                13,
            ),
        ).pack(
            side="left",
            padx=16,
            pady=(10, 0),
        )

        self.status = tk.Label(
            top,
            text="● Checking services",
            bg="#08111f",
            fg="#fbbf24",
            font=(
                "Segoe UI",
                11,
                "bold",
            ),
        )

        self.status.pack(
            side="right"
        )

        # =====================================================
        # MAIN LAYOUT
        # =====================================================

        main = tk.Frame(
            self.root,
            bg="#08111f",
            padx=28,
        )

        main.pack(
            fill="both",
            expand=True,
        )

        main.grid_rowconfigure(
            0,
            weight=1,
        )

        main.grid_columnconfigure(
            0,
            weight=1,
        )

        main.grid_columnconfigure(
            1,
            minsize=450,
        )

        left = tk.Frame(
            main,
            bg="#0d1a2c",
            padx=16,
            pady=16,
            highlightbackground="#1d3655",
            highlightthickness=1,
        )

        right_shell = tk.Frame(
            main,
            bg="#0d1a2c",
            width=450,
            highlightbackground="#1d3655",
            highlightthickness=1,
        )

        left.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 14),
        )

        right_shell.grid(
            row=0,
            column=1,
            sticky="nsew",
        )

        right_shell.grid_propagate(
            False
        )

        right_shell.grid_rowconfigure(
            0,
            weight=1,
        )

        right_shell.grid_columnconfigure(
            0,
            weight=1,
        )

        # =====================================================
        # SCROLLABLE SIDEBAR
        # =====================================================

        self.sidebar_canvas = tk.Canvas(
            right_shell,
            bg="#0d1a2c",
            highlightthickness=0,
            width=426,
        )

        sidebar_scrollbar = tk.Scrollbar(
            right_shell,
            orient="vertical",
            command=self.sidebar_canvas.yview,
        )

        self.sidebar_canvas.configure(
            yscrollcommand=sidebar_scrollbar.set
        )

        self.sidebar_canvas.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        sidebar_scrollbar.grid(
            row=0,
            column=1,
            sticky="ns",
        )

        right = tk.Frame(
            self.sidebar_canvas,
            bg="#0d1a2c",
            padx=22,
            pady=20,
        )

        self.sidebar_window = (
            self.sidebar_canvas.create_window(
                (0, 0),
                window=right,
                anchor="nw",
            )
        )

        right.bind(
            "<Configure>",
            lambda _event:
            self.sidebar_canvas.configure(
                scrollregion=(
                    self.sidebar_canvas.bbox(
                        "all"
                    )
                )
            ),
        )

        self.sidebar_canvas.bind(
            "<Configure>",
            lambda event:
            self.sidebar_canvas.itemconfigure(
                self.sidebar_window,
                width=event.width,
            ),
        )

        right.bind(
            "<Enter>",
            lambda _event:
            self.root.bind_all(
                "<MouseWheel>",
                self._scroll_sidebar,
            ),
        )

        right.bind(
            "<Leave>",
            lambda _event:
            self.root.unbind_all(
                "<MouseWheel>"
            ),
        )

        # =====================================================
        # VIDEO AREA
        # =====================================================

        self.video = tk.Label(
            left,
            bg="#050b14",
            text="Camera stopped",
            fg="#7f93ad",
            font=(
                "Segoe UI",
                16,
            ),
        )

        self.video.pack(
            fill="both",
            expand=True,
        )

        # =====================================================
        # CAMERA CONTROL BAR
        # =====================================================

        bar = tk.Frame(
            left,
            bg="#0d1a2c",
            pady=14,
        )

        bar.pack(
            fill="x"
        )

        self.start_btn = self._button(
            bar,
            "▶ Start Camera",
            self.toggle_camera,
            "#22c55e",
        )

        self.start_btn.pack(
            side="left"
        )

        self._button(
            bar,
            "Speak",
            self.speak,
            "#3b82f6",
        ).pack(
            side="left",
            padx=10,
        )

        self._button(
            bar,
            "Clear",
            self.clear,
            "#ef4444",
        ).pack(
            side="left"
        )

        self.conf = tk.Label(
            bar,
            text="Confidence: —",
            bg="#0d1a2c",
            fg="#93a9c4",
            font=(
                "Segoe UI",
                10,
            ),
        )

        self.conf.pack(
            side="right"
        )

        # =====================================================
        # RECOGNITION MODE
        # =====================================================

        mode_bar = tk.Frame(
            left,
            bg="#0d1a2c",
            pady=4,
        )

        mode_bar.pack(
            fill="x"
        )

        tk.Label(
            mode_bar,
            text="Recognition mode:",
            bg="#0d1a2c",
            fg="#93a9c4",
            font=(
                "Segoe UI",
                9,
                "bold",
            ),
        ).pack(
            side="left",
            padx=(0, 10),
        )

        self.alphabet_mode_btn = (
            self._button(
                mode_bar,
                "A–Z Alphabet",
                lambda:
                self.set_recognition_mode(
                    "alphabet"
                ),
                "#22c55e",
            )
        )

        self.alphabet_mode_btn.pack(
            side="left"
        )

        self.phrase_mode_btn = (
            self._button(
                mode_bar,
                "Dynamic Phrases",
                lambda:
                self.set_recognition_mode(
                    "phrase"
                ),
                "#334155",
            )
        )

        self.phrase_mode_btn.pack(
            side="left",
            padx=8,
        )

        self.mode_status = tk.Label(
            mode_bar,
            text="Alphabet mode",
            bg="#0d1a2c",
            fg="#6ee7ff",
            font=(
                "Segoe UI",
                9,
                "bold",
            ),
        )

        self.mode_status.pack(
            side="right"
        )

        # =====================================================
        # LIVE TRANSLATION
        # =====================================================

        tk.Label(
            right,
            text="LIVE TRANSLATION",
            bg="#0d1a2c",
            fg="#7f93ad",
            font=(
                "Segoe UI",
                10,
                "bold",
            ),
        ).pack(
            anchor="w"
        )

        self.current = tk.Label(
            right,
            text="—",
            bg="#0d1a2c",
            fg="#6ee7ff",
            font=(
                "Segoe UI",
                42,
                "bold",
            ),
        )

        self.current.pack(
            anchor="w",
            pady=(4, 6),
        )

        self.phrase_progress = tk.Label(
            right,
            text="",
            bg="#0d1a2c",
            fg="#93a9c4",
            font=(
                "Segoe UI",
                9,
            ),
        )

        self.phrase_progress.pack(
            anchor="w",
            pady=(0, 6),
        )

        tk.Label(
            right,
            text="Sentence / fingerspelling",
            bg="#0d1a2c",
            fg="#d7e4f5",
            font=(
                "Segoe UI",
                11,
                "bold",
            ),
        ).pack(
            anchor="w"
        )

        self.text = tk.Text(
            right,
            width=1,
            height=3,
            wrap="word",
            bg="#07101d",
            fg="white",
            insertbackground="white",
            relief="flat",
            font=(
                "Segoe UI",
                15,
            ),
            padx=12,
            pady=8,
        )

        self.text.pack(
            fill="x",
            pady=(8, 10),
        )

        # =====================================================
        # TEXT ACTIONS
        # =====================================================

        actions = tk.Frame(
            right,
            bg="#0d1a2c",
        )

        actions.pack(
            fill="x"
        )

        self._button(
            actions,
            "+ Space",
            self.add_space,
            "#334155",
        ).pack(
            side="left"
        )

        self._button(
            actions,
            "⌫ Delete",
            self.delete,
            "#334155",
        ).pack(
            side="left",
            padx=8,
        )

        self.smart_phrase_btn = (
            self._button(
                actions,
                "Smart Phrase",
                self.smart_phrase,
                "#8b5cf6",
            )
        )

        self.smart_phrase_btn.pack(
            side="left"
        )

        # =====================================================
        # DISTRIBUTED SERVICES
        # =====================================================

        service = tk.Frame(
            right,
            bg="#07101d",
            padx=10,
            pady=6,
        )

        service.pack(
            fill="x",
            pady=(10, 4),
        )

        tk.Label(
            service,
            text="DISTRIBUTED SERVICES",
            bg="#07101d",
            fg="#7f93ad",
            font=(
                "Segoe UI",
                9,
                "bold",
            ),
        ).grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
        )

        self.service_values = {}

        fields = [
            (
                "coordinator",
                "Coordinator",
            ),
            (
                "redis",
                "Redis",
            ),
            (
                "alphabet_workers",
                "Alphabet workers",
            ),
            (
                "phrase_workers",
                "Phrase workers",
            ),
            (
                "last_request_id",
                "Last request ID",
            ),
            (
                "last_worker_id",
                "Last worker ID",
            ),
            (
                "last_processing_ms",
                "Last processing",
            ),
            (
                "last_task_status",
                "Last task status",
            ),
        ]

        for row, (
            key,
            label,
        ) in enumerate(
            fields,
            start=1,
        ):
            tk.Label(
                service,
                text=label + ":",
                bg="#07101d",
                fg="#7f93ad",
                font=(
                    "Segoe UI",
                    8,
                ),
            ).grid(
                row=row,
                column=0,
                sticky="w",
                pady=1,
            )

            value = tk.Label(
                service,
                text="—",
                bg="#07101d",
                fg="#d7e4f5",
                font=(
                    "Segoe UI",
                    8,
                    "bold",
                ),
                anchor="e",
            )

            value.grid(
                row=row,
                column=1,
                sticky="e",
                padx=(8, 0),
                pady=1,
            )

            self.service_values[
                key
            ] = value

        service.grid_columnconfigure(
            1,
            weight=1,
        )

        # =====================================================
        # COMMON PHRASE BUTTONS
        # =====================================================

        tk.Label(
            right,
            text="COMMON PHRASES",
            bg="#0d1a2c",
            fg="#7f93ad",
            font=(
                "Segoe UI",
                10,
                "bold",
            ),
        ).pack(
            anchor="w",
            pady=(10, 4),
        )

        chips = tk.Frame(
            right,
            bg="#0d1a2c",
        )

        chips.pack(
            fill="x"
        )

        for index, phrase in enumerate(
            COMMON_PHRASES
        ):
            button = FlatButton(
                chips,
                text=phrase,
                command=lambda p=phrase:
                self.use_phrase(p),
                color="#1d3655",
                compact=True,
            )

            button.grid(
                row=index // 2,
                column=index % 2,
                sticky="ew",
                padx=3,
                pady=2,
            )

        chips.grid_columnconfigure(
            0,
            weight=1,
        )

        chips.grid_columnconfigure(
            1,
            weight=1,
        )

        tk.Label(
            right,
            text=(
                "Architecture: Webcam → MediaPipe → "
                "FastAPI → Redis → specialized Celery "
                "AI workers → text → speech"
            ),
            bg="#0d1a2c",
            fg="#617691",
            wraplength=360,
            justify="left",
            font=(
                "Segoe UI",
                9,
            ),
        ).pack(
            anchor="w",
            pady=(12, 8),
        )

    # =========================================================
    # BUTTON HELPERS
    # =========================================================

    def _button(
        self,
        parent,
        text,
        command,
        color,
    ):
        return FlatButton(
            parent,
            text=text,
            command=command,
            color=color,
        )

    def _scroll_sidebar(
        self,
        event,
    ):
        direction = (
            -1
            if event.delta > 0
            else 1
        )

        self.sidebar_canvas.yview_scroll(
            direction,
            "units",
        )

    # =========================================================
    # RECOGNITION MODE
    # =========================================================

    def set_recognition_mode(
        self,
        mode: str,
    ) -> None:

        if mode not in (
            "alphabet",
            "phrase",
        ):
            return

        self.recognition_mode = mode

        # Reset temporal state when switching modes
        self.phrase_buffer.clear()

        self.last_letter = None
        self.streak = 0

        self.conf.config(
            text="Confidence: —"
        )

        if mode == "alphabet":

            self.alphabet_mode_btn.normal_color = (
                "#22c55e"
            )

            self.alphabet_mode_btn.config(
                bg="#22c55e"
            )

            self.phrase_mode_btn.normal_color = (
                "#334155"
            )

            self.phrase_mode_btn.config(
                bg="#334155"
            )

            self.mode_status.config(
                text="Alphabet mode",
                fg="#6ee7ff",
            )

            self.phrase_progress.config(
                text=""
            )

            self.current.config(
                text="—",
                font=(
                    "Segoe UI",
                    42,
                    "bold",
                ),
            )

        else:

            self.alphabet_mode_btn.normal_color = (
                "#334155"
            )

            self.alphabet_mode_btn.config(
                bg="#334155"
            )

            self.phrase_mode_btn.normal_color = (
                "#8b5cf6"
            )

            self.phrase_mode_btn.config(
                bg="#8b5cf6"
            )

            self.mode_status.config(
                text="Dynamic phrase mode",
                fg="#c4b5fd",
            )

            self.phrase_progress.config(
                text="Show a phrase to the camera"
            )

            self.current.config(
                text="Ready",
                font=(
                    "Segoe UI",
                    24,
                    "bold",
                ),
            )

    # =========================================================
    # CAMERA
    # =========================================================

    def toggle_camera(
        self,
    ):

        if self.running:

            self.running = False

            self.camera.stop()

            self.video.config(
                image="",
                text="Camera stopped",
            )

            self.start_btn.normal_color = (
                "#22c55e"
            )

            self.start_btn.config(
                text="▶ Start Camera",
                bg="#22c55e",
            )

            self.phrase_buffer.clear()

            return

        try:

            self.camera = Camera(0)

            self.camera.start()

            self.running = True

            self.start_btn.normal_color = (
                "#ef4444"
            )

            self.start_btn.config(
                text="■ Stop Camera",
                bg="#ef4444",
            )

            self._loop()

        except Exception as exc:

            messagebox.showerror(
                "Camera error",
                str(exc),
            )

    # =========================================================
    # MAIN CAMERA LOOP
    # =========================================================

    def _loop(
        self,
    ):

        if not self.running:
            return

        (
            ok,
            frame,
            alphabet_features,
            phrase_features,
        ) = self.camera.read()

        if ok:

            # -----------------------------------------------
            # Display webcam
            # -----------------------------------------------

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            img = Image.fromarray(
                rgb
            )

            w = max(
                480,
                self.root.winfo_width()
                - 600,
            )

            h = max(
                300,
                self.root.winfo_height()
                - 400,
            )

            img.thumbnail(
                (
                    w,
                    h,
                )
            )

            self.photo = (
                ImageTk.PhotoImage(
                    img
                )
            )

            self.video.config(
                image=self.photo,
                text="",
            )

            # -----------------------------------------------
            # Recognition
            # -----------------------------------------------

            if (
                self.recognition_mode
                == "alphabet"
            ):
                self._process_alphabet_frame(
                    alphabet_features
                )

            elif (
                self.recognition_mode
                == "phrase"
            ):
                self._process_phrase_frame(
                    phrase_features
                )

        self._drain_results()

        self.root.after(
            20,
            self._loop,
        )

    # =========================================================
    # ALPHABET CAMERA PROCESSING
    # =========================================================

    def _process_alphabet_frame(
        self,
        features,
    ) -> None:

        if not features:
            return

        now = time.time()

        if (
            not self.alphabet_inflight
            and (
                now
                - self.last_alphabet_submit
                > 0.12
            )
        ):
            self.alphabet_inflight = True

            self.last_alphabet_submit = now

            threading.Thread(
                target=self._predict_alphabet,
                args=(
                    features,
                ),
                daemon=True,
            ).start()

    # =========================================================
    # PHRASE CAMERA PROCESSING
    # =========================================================

    def _process_phrase_frame(
        self,
        phrase_features,
    ) -> None:

        if phrase_features is None:

            self.phrase_progress.config(
                text=(
                    "No hand detected — "
                    "waiting for sign"
                )
            )

            return

        # Avoid collecting another phrase while the
        # current sequence is being classified.
        if self.phrase_inflight:
            self.phrase_progress.config(
                text="Recognizing phrase…"
            )

            return

        try:

            ready = self.phrase_buffer.add(
                phrase_features
            )

        except Exception as exc:

            self.phrase_buffer.clear()

            self.phrase_progress.config(
                text=f"Phrase input error: {exc}"
            )

            return

        self.phrase_progress.config(
            text=(
                "Collecting phrase: "
                f"{self.phrase_buffer.size}/"
                f"{self.phrase_buffer.sequence_length}"
                " frames"
            )
        )

        if not ready:
            return

        now = time.time()

        if (
            now
            - self.last_phrase_submit
            < 0.5
        ):
            return

        sequence = (
            self.phrase_buffer.pop_sequence()
        )

        if sequence is None:
            return

        self.last_phrase_submit = now
        self.phrase_inflight = True

        self.phrase_progress.config(
            text="Recognizing phrase…"
        )

        threading.Thread(
            target=self._predict_phrase,
            args=(
                sequence,
            ),
            daemon=True,
        ).start()

    # =========================================================
    # ALPHABET DISTRIBUTED REQUEST
    # =========================================================

    def _predict_alphabet(
        self,
        features,
    ):

        request_id = new_request_id()

        started = time.perf_counter()

        try:

            result = post_alphabet(
                features,
                request_id,
            )

            log_event(
                logger,
                "prediction_received",
                component="client",
                request_id=result.get(
                    "request_id",
                    request_id,
                ),
                worker_id=result.get(
                    "worker_id"
                ),
                task_name=(
                    "workers.alphabet_tasks."
                    "predict_alphabet"
                ),
                processing_ms=(
                    time.perf_counter()
                    - started
                )
                * 1000,
                success=True,
            )

            self.result_queue.put(
                (
                    "alphabet_ok",
                    result,
                )
            )

        except Exception as exc:

            log_event(
                logger,
                "prediction_failed",
                component="client",
                request_id=request_id,
                task_name=(
                    "workers.alphabet_tasks."
                    "predict_alphabet"
                ),
                processing_ms=(
                    time.perf_counter()
                    - started
                )
                * 1000,
                success=False,
                level=logging.ERROR,
            )

            self.result_queue.put(
                (
                    "alphabet_error",
                    {
                        "message": str(exc),
                        "request_id":
                        request_id,
                    },
                )
            )

        finally:

            self.alphabet_inflight = False

    # =========================================================
    # PHRASE DISTRIBUTED REQUEST
    # =========================================================

    def _predict_phrase(
        self,
        sequence,
    ):

        request_id = new_request_id()

        started = time.perf_counter()

        try:

            result = post_phrase_sequence(
                sequence,
                request_id,
            )

            log_event(
                logger,
                "phrase_prediction_received",
                component="client",
                request_id=result.get(
                    "request_id",
                    request_id,
                ),
                worker_id=result.get(
                    "worker_id"
                ),
                task_name=(
                    "workers.phrase_tasks."
                    "predict_phrase"
                ),
                processing_ms=(
                    time.perf_counter()
                    - started
                )
                * 1000,
                success=True,
            )

            self.result_queue.put(
                (
                    "phrase_ok",
                    result,
                )
            )

        except Exception as exc:

            log_event(
                logger,
                "phrase_prediction_failed",
                component="client",
                request_id=request_id,
                task_name=(
                    "workers.phrase_tasks."
                    "predict_phrase"
                ),
                processing_ms=(
                    time.perf_counter()
                    - started
                )
                * 1000,
                success=False,
                level=logging.ERROR,
            )

            self.result_queue.put(
                (
                    "phrase_error",
                    {
                        "message": str(exc),
                        "request_id":
                        request_id,
                    },
                )
            )

        finally:

            self.phrase_inflight = False

    # =========================================================
    # RESULT QUEUE
    # =========================================================

    def _drain_results(
        self,
    ):

        try:

            while True:

                (
                    kind,
                    payload,
                ) = (
                    self.result_queue
                    .get_nowait()
                )

                if (
                    kind
                    == "alphabet_error"
                ):

                    self._record_task_result(
                        request_id=payload.get(
                            "request_id"
                        ),
                        task_status="error",
                    )

                    self.status.config(
                        text=(
                            "● Alphabet worker "
                            "unavailable"
                        ),
                        fg="#ef4444",
                    )

                    continue

                if (
                    kind
                    == "phrase_error"
                ):

                    self._record_task_result(
                        request_id=payload.get(
                            "request_id"
                        ),
                        task_status="error",
                    )

                    self.status.config(
                        text=(
                            "● Phrase worker "
                            "unavailable"
                        ),
                        fg="#ef4444",
                    )

                    self.phrase_progress.config(
                        text=(
                            "Phrase recognition "
                            "failed"
                        )
                    )

                    continue

                if (
                    kind
                    == "alphabet_ok"
                ):

                    self._handle_alphabet_result(
                        payload
                    )

                elif (
                    kind
                    == "phrase_ok"
                ):

                    self._handle_phrase_result(
                        payload
                    )

        except queue.Empty:
            pass

    # =========================================================
    # ALPHABET RESULT
    # =========================================================

    def _handle_alphabet_result(
        self,
        payload,
    ) -> None:

        self._record_task_result(
            request_id=payload.get(
                "request_id"
            ),
            worker_id=payload.get(
                "worker_id"
            ),
            processing_ms=payload.get(
                "inference_ms"
            ),
            task_status=(
                "failed"
                if payload.get(
                    "status"
                )
                == "model_missing"
                else "completed"
            ),
        )

        if (
            payload.get(
                "status"
            )
            == "model_missing"
        ):

            self.status.config(
                text=(
                    "● Train alphabet "
                    "model first"
                ),
                fg="#fbbf24",
            )

            return

        letter = payload.get(
            "letter"
        )

        confidence = payload.get(
            "confidence"
        )

        self.conf.config(
            text=(
                f"Confidence: "
                f"{confidence:.1%}"
                if confidence is not None
                else "Confidence: —"
            )
        )

        self.current.config(
            text=letter or "…",
            font=(
                "Segoe UI",
                42,
                "bold",
            ),
        )

        if letter:

            self._stabilize(
                letter
            )

    # =========================================================
    # PHRASE RESULT
    # =========================================================

    def _handle_phrase_result(
        self,
        payload,
    ) -> None:

        self._record_task_result(
            request_id=payload.get(
                "request_id"
            ),
            worker_id=payload.get(
                "worker_id"
            ),
            processing_ms=payload.get(
                "inference_ms"
            ),
            task_status=(
                "failed"
                if payload.get(
                    "status"
                )
                == "model_missing"
                else "completed"
            ),
        )

        status = payload.get(
            "status"
        )

        confidence = payload.get(
            "confidence"
        )

        predicted_label = payload.get(
            "predicted_label"
        )

        accepted_phrase = payload.get(
            "phrase"
        )

        self.conf.config(
            text=(
                f"Confidence: "
                f"{confidence:.1%}"
                if confidence is not None
                else "Confidence: —"
            )
        )

        if status == "model_missing":

            self.current.config(
                text="Model unavailable",
                font=(
                    "Segoe UI",
                    18,
                    "bold",
                ),
            )

            self.phrase_progress.config(
                text="Train phrase model first"
            )

            return

        if (
            status == "predicted"
            and accepted_phrase
        ):

            self.current.config(
                text=accepted_phrase,
                font=(
                    "Segoe UI",
                    26,
                    "bold",
                ),
            )

            self.phrase_progress.config(
                text=(
                    "Phrase recognized ✓"
                )
            )

            self._emit_phrase(
                accepted_phrase
            )

            return

        # Prediction existed but confidence was
        # below threshold.
        self.current.config(
            text=(
                predicted_label
                or "Uncertain"
            ),
            font=(
                "Segoe UI",
                24,
                "bold",
            ),
        )

        self.phrase_progress.config(
            text=(
                "Low confidence — "
                "please sign again"
            )
        )

    # =========================================================
    # PHRASE OUTPUT / DEBOUNCE
    # =========================================================

    def _emit_phrase(
        self,
        phrase: str,
    ) -> None:

        now = time.time()

        # Avoid continuously repeating the exact
        # same phrase if the signer keeps holding it.
        if (
            phrase
            == self.last_phrase
            and (
                now
                - self.last_phrase_emit
                < 3.0
            )
        ):
            return

        self.last_phrase = phrase
        self.last_phrase_emit = now

        existing = self.buffer.text.strip()

        if existing:
            new_text = (
                existing
                + " "
                + phrase
            )
        else:
            new_text = phrase

        self.buffer.set_text(
            new_text
        )

        self._sync_text()

        # Dynamic sign → speech
        self.voice.speak(
            phrase
        )

    # =========================================================
    # ALPHABET STABILIZATION
    # =========================================================

    def _stabilize(
        self,
        letter,
    ):

        if letter == self.last_letter:

            self.streak += 1

        else:

            self.last_letter = letter
            self.streak = 1

        if (
            self.streak >= 3
            and (
                time.time()
                - self.last_emit
                > 1.15
            )
        ):

            self.buffer.add_letter(
                letter
            )

            self._sync_text()

            self.last_emit = (
                time.time()
            )

            self.streak = 0

    # =========================================================
    # TEXT BUFFER
    # =========================================================

    def _sync_text(
        self,
    ):

        self.text.delete(
            "1.0",
            "end",
        )

        self.text.insert(
            "1.0",
            self.buffer.text,
        )

    def _pull_text(
        self,
    ):

        self.buffer.set_text(
            self.text.get(
                "1.0",
                "end",
            ).strip()
        )

    def add_space(
        self,
    ):

        self._pull_text()
        self.buffer.space()
        self._sync_text()

    def delete(
        self,
    ):

        self._pull_text()
        self.buffer.delete()
        self._sync_text()

    def clear(
        self,
    ):

        self.buffer.clear()

        self.phrase_buffer.clear()

        self.last_letter = None
        self.streak = 0
        self.last_phrase = None

        self._sync_text()

        self.current.config(
            text="—"
        )

        self.conf.config(
            text="Confidence: —"
        )

        if (
            self.recognition_mode
            == "phrase"
        ):
            self.phrase_progress.config(
                text="Show a phrase to the camera"
            )

        else:
            self.phrase_progress.config(
                text=""
            )

    def speak(
        self,
    ):

        self._pull_text()

        self.voice.speak(
            self.buffer.text
        )

    def use_phrase(
        self,
        phrase,
    ):

        self.buffer.set_text(
            phrase
        )

        self._sync_text()

        self.voice.speak(
            phrase
        )

    # =========================================================
    # SMART PHRASE TEXT NORMALIZATION
    # =========================================================

    def smart_phrase(
        self,
    ):

        self._pull_text()

        raw = self.buffer.text

        if not raw:

            self.current.config(
                text="Enter text first"
            )

            return

        self.smart_phrase_btn.config(
            text="Processing…"
        )

        def work():

            request_id = (
                new_request_id()
            )

            started = (
                time.perf_counter()
            )

            try:

                result = (
                    post_phrase_text(
                        raw,
                        request_id,
                    )
                )

                log_event(
                    logger,
                    "normalization_received",
                    component="client",
                    request_id=result.get(
                        "request_id",
                        request_id,
                    ),
                    worker_id=result.get(
                        "worker_id"
                    ),
                    task_name=(
                        "workers.phrase_tasks."
                        "normalize_phrase"
                    ),
                    processing_ms=(
                        time.perf_counter()
                        - started
                    )
                    * 1000,
                    success=True,
                )

                self.root.after(
                    0,
                    lambda result=result:
                    self._apply_phrase_result(
                        result
                    ),
                )

            except Exception as exc:

                log_event(
                    logger,
                    "normalization_failed",
                    component="client",
                    request_id=request_id,
                    task_name=(
                        "workers.phrase_tasks."
                        "normalize_phrase"
                    ),
                    processing_ms=(
                        time.perf_counter()
                        - started
                    )
                    * 1000,
                    success=False,
                    level=logging.ERROR,
                )

                message = str(
                    exc
                )

                self.root.after(
                    0,
                    lambda
                    request_id=request_id,
                    message=message:
                    self._apply_phrase_error(
                        request_id,
                        message,
                    ),
                )

        threading.Thread(
            target=work,
            daemon=True,
        ).start()

    def _apply_phrase_result(
        self,
        result,
    ):

        self._apply_phrase(
            result["text"]
        )

        self.smart_phrase_btn.config(
            text="Smart Phrase"
        )

        self.current.config(
            text=(
                "✓ Phrase matched"
                if result.get(
                    "matched"
                )
                else "✓ Phrase ready"
            ),
            font=(
                "Segoe UI",
                18,
                "bold",
            ),
        )

        self.voice.speak(
            result["text"]
        )

        self._record_task_result(
            request_id=result.get(
                "request_id"
            ),
            worker_id=result.get(
                "worker_id"
            ),
            processing_ms=result.get(
                "inference_ms"
            ),
            task_status="completed",
        )

    def _apply_phrase_error(
        self,
        request_id,
        message,
    ):

        self.smart_phrase_btn.config(
            text="Smart Phrase"
        )

        self._record_task_result(
            request_id=request_id,
            task_status="error",
        )

        messagebox.showwarning(
            "Phrase worker",
            message,
        )

    def _apply_phrase(
        self,
        text,
    ):

        self.buffer.set_text(
            text
        )

        self._sync_text()

    # =========================================================
    # DISTRIBUTED TASK STATUS
    # =========================================================

    def _record_task_result(
        self,
        *,
        request_id=None,
        worker_id=None,
        processing_ms=None,
        task_status=None,
    ):

        if request_id:

            self.last_request_id = (
                request_id
            )

            self.status_poller.set_last_request_id(
                request_id
            )

        if worker_id:

            self.last_worker_id = (
                worker_id
            )

        if processing_ms is not None:

            self.last_processing_ms = (
                processing_ms
            )

        if task_status:

            self.last_task_status = (
                task_status
            )

        self._render_last_task_fields()

    def _render_last_task_fields(
        self,
    ):

        self.service_values[
            "last_request_id"
        ].config(
            text=(
                self.last_request_id
                or "—"
            )
        )

        self.service_values[
            "last_worker_id"
        ].config(
            text=(
                self.last_worker_id
                or "—"
            )
        )

        processing = (
            f"{self.last_processing_ms:.1f} ms"
            if (
                self.last_processing_ms
                is not None
            )
            else "—"
        )

        self.service_values[
            "last_processing_ms"
        ].config(
            text=processing
        )

        self.service_values[
            "last_task_status"
        ].config(
            text=(
                self.last_task_status
                or "—"
            )
        )

    # =========================================================
    # SERVICE STATUS POLLING
    # =========================================================

    def _drain_service_status(
        self,
    ):

        try:

            while True:

                snapshot = (
                    self.status_queue
                    .get_nowait()
                )

                self._apply_service_status(
                    snapshot
                )

        except queue.Empty:
            pass

        if not self.closing:

            self.root.after(
                250,
                self._drain_service_status,
            )

    def _apply_service_status(
        self,
        snapshot: ServiceStatus,
    ):

        coordinator_text = (
            "Online"
            if snapshot.coordinator_online
            else "Offline"
        )

        redis_text = (
            "Online"
            if snapshot.redis_online
            else "Offline"
        )

        self.service_values[
            "coordinator"
        ].config(
            text=coordinator_text,
            fg=(
                "#22c55e"
                if snapshot.coordinator_online
                else "#ef4444"
            ),
        )

        self.service_values[
            "redis"
        ].config(
            text=redis_text,
            fg=(
                "#22c55e"
                if snapshot.redis_online
                else "#ef4444"
            ),
        )

        self.service_values[
            "alphabet_workers"
        ].config(
            text=str(
                snapshot
                .alphabet_workers_online
            )
        )

        self.service_values[
            "phrase_workers"
        ].config(
            text=str(
                snapshot
                .phrase_workers_online
            )
        )

        if snapshot.last_request_id:

            self.last_request_id = (
                snapshot.last_request_id
            )

        if snapshot.last_worker_id:

            self.last_worker_id = (
                snapshot.last_worker_id
            )

        if (
            snapshot.last_processing_ms
            is not None
        ):

            self.last_processing_ms = (
                snapshot
                .last_processing_ms
            )

        if snapshot.last_task_status:

            self.last_task_status = (
                snapshot
                .last_task_status
            )

        self._render_last_task_fields()

        if (
            not snapshot
            .coordinator_online
        ):

            self.status.config(
                text="● Coordinator offline",
                fg="#ef4444",
            )

        elif (
            not snapshot.redis_online
        ):

            self.status.config(
                text=(
                    "● API online / "
                    "Redis offline"
                ),
                fg="#fbbf24",
            )

        else:

            self.status.config(
                text=(
                    "● Distributed "
                    "services online"
                ),
                fg="#22c55e",
            )

    # =========================================================
    # CLEAN SHUTDOWN
    # =========================================================

    def close(
        self,
    ):

        self.closing = True

        self.status_poller.stop()

        self.running = False

        try:
            self.camera.stop()

        except Exception:
            pass

        self.root.destroy()


# =============================================================
# MAIN
# =============================================================

def main() -> None:

    root = tk.Tk()

    app = ASLApp(
        root
    )

    root.protocol(
        "WM_DELETE_WINDOW",
        app.close,
    )

    root.mainloop()


if __name__ == "__main__":
    main()