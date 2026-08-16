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
from client.api_client import post_alphabet, post_phrase_text
from client.camera import Camera
from client.voice_service import VoiceService
from client.service_status import ServiceStatus, StatusPoller
from config import settings
from observability import get_logger, log_event, new_request_id

API = settings.api_url
logger = get_logger("client")
COMMON_PHRASES = ['Hello', 'Thank you', 'Good morning', 'Please', 'Yes', 'No', 'Help me', 'Sorry', 'I love you', 'How are you?', 'Nice to meet you', 'Goodbye']


class FlatButton(tk.Label):
    """A consistently coloured button on macOS, Windows, and Linux."""

    def __init__(self, parent, *, text, command, color, compact=False):
        self.command = command
        self.normal_color = color
        super().__init__(
            parent,
            text=text,
            bg=color,
            fg='white',
            cursor='hand2',
            font=('Segoe UI', 9 if compact else 10, 'bold'),
            padx=8 if compact else 14,
            pady=4 if compact else 9,
            takefocus=True,
        )
        self.bind('<Button-1>', self._invoke)
        self.bind('<Return>', self._invoke)
        self.bind('<space>', self._invoke)
        self.bind('<Enter>', lambda _event: self.config(bg=self._lighter(color)))
        self.bind('<Leave>', lambda _event: self.config(bg=self.normal_color))

    def _invoke(self, _event=None):
        self.command()

    @staticmethod
    def _lighter(color):
        red, green, blue = (int(color[index:index + 2], 16) for index in (1, 3, 5))
        return f'#{min(red + 24, 255):02x}{min(green + 24, 255):02x}{min(blue + 24, 255):02x}'

class ASLApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title('Distributed ASL-to-Speech Translation System')
        self.root.geometry('1280x780')
        self.root.minsize(1080, 700)
        self.root.configure(bg='#08111f')
        self.camera = Camera(0)
        self.buffer = AlphabetBuffer()
        self.voice = VoiceService()
        self.running = False
        self.inflight = False
        self.last_submit = 0.0
        self.last_letter = None
        self.streak = 0
        self.last_emit = 0.0
        self.result_queue: queue.Queue = queue.Queue()
        self.status_queue: queue.Queue = queue.Queue(maxsize=1)
        self.last_request_id = None
        self.last_worker_id = None
        self.last_processing_ms = None
        self.last_task_status = None
        self.closing = False
        self._build_ui()
        self.status_poller = StatusPoller(self.status_queue, api_url=API)
        self.status_poller.start()
        self.root.after(200, self._drain_service_status)

    def _build_ui(self) -> None:
        top = tk.Frame(self.root, bg='#08111f', padx=28, pady=22)
        top.pack(fill='x')
        tk.Label(top, text='Distributed ASL-to-Speech', bg='#08111f', fg='#6ee7ff', font=('Segoe UI', 27, 'bold')).pack(side='left')
        tk.Label(top, text='Translation System', bg='#08111f', fg='#d7e4f5', font=('Segoe UI', 13)).pack(side='left', padx=16, pady=(10,0))
        self.status = tk.Label(top, text='● checking services', bg='#08111f', fg='#fbbf24', font=('Segoe UI', 11, 'bold'))
        self.status.pack(side='right')

        main = tk.Frame(self.root, bg='#08111f', padx=28)
        main.pack(fill='both', expand=True)
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, minsize=450)
        left = tk.Frame(main, bg='#0d1a2c', padx=16, pady=16, highlightbackground='#1d3655', highlightthickness=1)
        right_shell = tk.Frame(
            main, bg='#0d1a2c', width=450,
            highlightbackground='#1d3655', highlightthickness=1,
        )
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 14))
        right_shell.grid(row=0, column=1, sticky='nsew')
        right_shell.grid_propagate(False)
        right_shell.grid_rowconfigure(0, weight=1)
        right_shell.grid_columnconfigure(0, weight=1)

        self.sidebar_canvas = tk.Canvas(
            right_shell, bg='#0d1a2c', highlightthickness=0, width=426,
        )
        sidebar_scrollbar = tk.Scrollbar(
            right_shell, orient='vertical', command=self.sidebar_canvas.yview,
        )
        self.sidebar_canvas.configure(yscrollcommand=sidebar_scrollbar.set)
        self.sidebar_canvas.grid(row=0, column=0, sticky='nsew')
        sidebar_scrollbar.grid(row=0, column=1, sticky='ns')
        right = tk.Frame(self.sidebar_canvas, bg='#0d1a2c', padx=22, pady=20)
        self.sidebar_window = self.sidebar_canvas.create_window(
            (0, 0), window=right, anchor='nw',
        )
        right.bind(
            '<Configure>',
            lambda _event: self.sidebar_canvas.configure(
                scrollregion=self.sidebar_canvas.bbox('all')
            ),
        )
        self.sidebar_canvas.bind(
            '<Configure>',
            lambda event: self.sidebar_canvas.itemconfigure(
                self.sidebar_window, width=event.width
            ),
        )
        right.bind('<Enter>', lambda _event: self.root.bind_all('<MouseWheel>', self._scroll_sidebar))
        right.bind('<Leave>', lambda _event: self.root.unbind_all('<MouseWheel>'))

        self.video = tk.Label(left, bg='#050b14', text='Camera stopped', fg='#7f93ad', font=('Segoe UI', 16))
        self.video.pack(fill='both', expand=True)
        bar = tk.Frame(left, bg='#0d1a2c', pady=14)
        bar.pack(fill='x')
        self.start_btn = self._button(bar, '▶ Start Camera', self.toggle_camera, '#22c55e')
        self.start_btn.pack(side='left')
        self._button(bar, 'Speak', self.speak, '#3b82f6').pack(side='left', padx=10)
        self._button(bar, 'Clear', self.clear, '#ef4444').pack(side='left')
        self.conf = tk.Label(bar, text='Confidence: —', bg='#0d1a2c', fg='#93a9c4', font=('Segoe UI', 10))
        self.conf.pack(side='right')

        tk.Label(right, text='LIVE TRANSLATION', bg='#0d1a2c', fg='#7f93ad', font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        self.current = tk.Label(right, text='—', bg='#0d1a2c', fg='#6ee7ff', font=('Segoe UI', 42, 'bold'))
        self.current.pack(anchor='w', pady=(4, 6))
        tk.Label(right, text='Sentence / fingerspelling', bg='#0d1a2c', fg='#d7e4f5', font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        self.text = tk.Text(right, width=1, height=3, wrap='word', bg='#07101d', fg='white', insertbackground='white', relief='flat', font=('Segoe UI', 15), padx=12, pady=8)
        self.text.pack(fill='x', pady=(8, 10))
        actions = tk.Frame(right, bg='#0d1a2c')
        actions.pack(fill='x')
        self._button(actions, '+ Space', self.add_space, '#334155').pack(side='left')
        self._button(actions, '⌫ Delete', self.delete, '#334155').pack(side='left', padx=8)
        self.smart_phrase_btn = self._button(
            actions, 'Smart Phrase', self.smart_phrase, '#8b5cf6'
        )
        self.smart_phrase_btn.pack(side='left')

        service = tk.Frame(right, bg='#07101d', padx=10, pady=6)
        service.pack(fill='x', pady=(10, 4))
        tk.Label(service, text='DISTRIBUTED SERVICES', bg='#07101d', fg='#7f93ad',
                 font=('Segoe UI', 9, 'bold')).grid(row=0, column=0, columnspan=2, sticky='w')
        self.service_values = {}
        fields = [
            ('coordinator', 'Coordinator'), ('redis', 'Redis'),
            ('alphabet_workers', 'Alphabet workers'), ('phrase_workers', 'Phrase workers'),
            ('last_request_id', 'Last request ID'), ('last_worker_id', 'Last worker ID'),
            ('last_processing_ms', 'Last processing'), ('last_task_status', 'Last task status'),
        ]
        for row, (key, label) in enumerate(fields, start=1):
            tk.Label(service, text=label + ':', bg='#07101d', fg='#7f93ad',
                     font=('Segoe UI', 8)).grid(row=row, column=0, sticky='w', pady=1)
            value = tk.Label(service, text='—', bg='#07101d', fg='#d7e4f5',
                             font=('Segoe UI', 8, 'bold'), anchor='e')
            value.grid(row=row, column=1, sticky='e', padx=(8, 0), pady=1)
            self.service_values[key] = value
        service.grid_columnconfigure(1, weight=1)

        tk.Label(right, text='COMMON PHRASES', bg='#0d1a2c', fg='#7f93ad', font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(10, 4))
        chips = tk.Frame(right, bg='#0d1a2c')
        chips.pack(fill='x')
        for index, phrase in enumerate(COMMON_PHRASES):
            b = FlatButton(
                chips, text=phrase, command=lambda p=phrase: self.use_phrase(p),
                color='#1d3655', compact=True,
            )
            b.grid(row=index // 2, column=index % 2, sticky='ew', padx=3, pady=2)
        chips.grid_columnconfigure(0, weight=1)
        chips.grid_columnconfigure(1, weight=1)
        tk.Label(right, text='Architecture: Webcam → MediaPipe → FastAPI → Redis → specialized Celery AI workers → text → speech', bg='#0d1a2c', fg='#617691', wraplength=360, justify='left', font=('Segoe UI', 9)).pack(anchor='w', pady=(12, 8))

    def _button(self, parent, text, command, color):
        return FlatButton(parent, text=text, command=command, color=color)

    def _scroll_sidebar(self, event):
        direction = -1 if event.delta > 0 else 1
        self.sidebar_canvas.yview_scroll(direction, 'units')

    def toggle_camera(self):
        if self.running:
            self.running = False
            self.camera.stop()
            self.video.config(image='', text='Camera stopped')
            self.start_btn.normal_color = '#22c55e'
            self.start_btn.config(text='▶ Start Camera', bg='#22c55e')
            return
        try:
            self.camera = Camera(0)
            self.camera.start()
            self.running = True
            self.start_btn.normal_color = '#ef4444'
            self.start_btn.config(text='■ Stop Camera', bg='#ef4444')
            self._loop()
        except Exception as exc:
            messagebox.showerror('Camera error', str(exc))

    def _loop(self):
        if not self.running:
            return
        ok, frame, features = self.camera.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            # Reserve horizontal space for the sidebar and vertical space for
            # the title/control bars. Using the image label's requested size
            # here creates a feedback loop in which each frame enlarges it.
            w = max(480, self.root.winfo_width() - 600)
            h = max(300, self.root.winfo_height() - 400)
            img.thumbnail((w, h))
            self.photo = ImageTk.PhotoImage(img)
            self.video.config(image=self.photo, text='')
            now = time.time()
            if features and not self.inflight and now - self.last_submit > 0.12:
                self.inflight = True
                self.last_submit = now
                threading.Thread(target=self._predict, args=(features,), daemon=True).start()
        self._drain_results()
        self.root.after(20, self._loop)

    def _predict(self, features):
        request_id = new_request_id()
        started = time.perf_counter()
        try:
            result = post_alphabet(features, request_id)
            log_event(
                logger, 'prediction_received', component='client',
                request_id=result.get('request_id', request_id),
                worker_id=result.get('worker_id'),
                task_name='workers.alphabet_tasks.predict_alphabet',
                processing_ms=(time.perf_counter() - started) * 1000,
                success=True,
            )
            self.result_queue.put(('ok', result))
        except Exception as exc:
            log_event(
                logger, 'prediction_failed', component='client',
                request_id=request_id,
                task_name='workers.alphabet_tasks.predict_alphabet',
                processing_ms=(time.perf_counter() - started) * 1000,
                success=False, level=logging.ERROR,
            )
            self.result_queue.put(('err', {'message': str(exc), 'request_id': request_id}))
        finally:
            self.inflight = False

    def _drain_results(self):
        try:
            while True:
                kind, payload = self.result_queue.get_nowait()
                if kind == 'err':
                    self._record_task_result(
                        request_id=payload.get('request_id'), task_status='error'
                    )
                    self.status.config(text='● worker unavailable', fg='#ef4444')
                    continue
                self._record_task_result(
                    request_id=payload.get('request_id'),
                    worker_id=payload.get('worker_id'),
                    processing_ms=payload.get('inference_ms'),
                    task_status='failed' if payload.get('status') == 'model_missing' else 'completed',
                )
                if payload.get('status') == 'model_missing':
                    self.status.config(text='● Train alphabet model first', fg='#fbbf24')
                letter = payload.get('letter')
                confidence = payload.get('confidence')
                self.conf.config(text=f"Confidence: {confidence:.1%}" if confidence is not None else 'Confidence: —')
                self.current.config(
                    text=letter or '…', font=('Segoe UI', 42, 'bold')
                )
                if letter:
                    self._stabilize(letter)
        except queue.Empty:
            pass

    def _stabilize(self, letter):
        if letter == self.last_letter:
            self.streak += 1
        else:
            self.last_letter, self.streak = letter, 1
        # The worker classifies individual landmark vectors; the UI adds a light debounce.
        if self.streak >= 3 and time.time() - self.last_emit > 1.15:
            self.buffer.add_letter(letter)
            self._sync_text()
            self.last_emit = time.time()
            self.streak = 0

    def _sync_text(self):
        self.text.delete('1.0', 'end')
        self.text.insert('1.0', self.buffer.text)

    def _pull_text(self):
        self.buffer.set_text(self.text.get('1.0', 'end').strip())

    def add_space(self):
        self._pull_text(); self.buffer.space(); self._sync_text()
    def delete(self):
        self._pull_text(); self.buffer.delete(); self._sync_text()
    def clear(self):
        self.buffer.clear(); self._sync_text(); self.current.config(text='—')
    def speak(self):
        self._pull_text(); self.voice.speak(self.buffer.text)
    def use_phrase(self, phrase):
        self.buffer.set_text(phrase); self._sync_text(); self.voice.speak(phrase)
    def smart_phrase(self):
        self._pull_text()
        raw = self.buffer.text
        if not raw:
            self.current.config(text='Enter text first')
            return
        self.smart_phrase_btn.config(text='Processing…')
        def work():
            request_id = new_request_id()
            started = time.perf_counter()
            try:
                result = post_phrase_text(raw, request_id); normalized = result['text']
                log_event(
                    logger, 'normalization_received', component='client',
                    request_id=result.get('request_id', request_id),
                    worker_id=result.get('worker_id'),
                    task_name='workers.phrase_tasks.normalize_phrase',
                    processing_ms=(time.perf_counter() - started) * 1000,
                    success=True,
                )
                self.root.after(0, lambda result=result:self._apply_phrase_result(result))
            except Exception as exc:
                log_event(
                    logger, 'normalization_failed', component='client',
                    request_id=request_id,
                    task_name='workers.phrase_tasks.normalize_phrase',
                    processing_ms=(time.perf_counter() - started) * 1000,
                    success=False, level=logging.ERROR,
                )
                message = str(exc)
                self.root.after(
                    0,
                    lambda request_id=request_id, message=message:
                    self._apply_phrase_error(request_id, message),
                )
        threading.Thread(target=work, daemon=True).start()
    def _apply_phrase_result(self, result):
        self._apply_phrase(result['text'])
        self.smart_phrase_btn.config(text='Smart Phrase')
        self.current.config(
            text='✓ Phrase matched' if result.get('matched') else '✓ Phrase ready',
            font=('Segoe UI', 18, 'bold'),
        )
        self.voice.speak(result['text'])
        self._record_task_result(
            request_id=result.get('request_id'), worker_id=result.get('worker_id'),
            processing_ms=result.get('inference_ms'), task_status='completed',
        )
    def _apply_phrase_error(self, request_id, message):
        self.smart_phrase_btn.config(text='Smart Phrase')
        self._record_task_result(request_id=request_id, task_status='error')
        messagebox.showwarning('Phrase worker', message)
    def _apply_phrase(self, text):
        self.buffer.set_text(text); self._sync_text()

    def _record_task_result(
        self, *, request_id=None, worker_id=None, processing_ms=None, task_status=None
    ):
        if request_id:
            self.last_request_id = request_id
            self.status_poller.set_last_request_id(request_id)
        if worker_id:
            self.last_worker_id = worker_id
        if processing_ms is not None:
            self.last_processing_ms = processing_ms
        if task_status:
            self.last_task_status = task_status
        self._render_last_task_fields()

    def _render_last_task_fields(self):
        self.service_values['last_request_id'].config(
            text=self.last_request_id or '—'
        )
        self.service_values['last_worker_id'].config(
            text=self.last_worker_id or '—'
        )
        processing = (
            f'{self.last_processing_ms:.1f} ms'
            if self.last_processing_ms is not None else '—'
        )
        self.service_values['last_processing_ms'].config(text=processing)
        self.service_values['last_task_status'].config(
            text=self.last_task_status or '—'
        )

    def _drain_service_status(self):
        try:
            while True:
                snapshot = self.status_queue.get_nowait()
                self._apply_service_status(snapshot)
        except queue.Empty:
            pass
        if not self.closing:
            self.root.after(250, self._drain_service_status)

    def _apply_service_status(self, snapshot: ServiceStatus):
        coordinator_text = 'Online' if snapshot.coordinator_online else 'Offline'
        redis_text = 'Online' if snapshot.redis_online else 'Offline'
        self.service_values['coordinator'].config(
            text=coordinator_text, fg='#22c55e' if snapshot.coordinator_online else '#ef4444'
        )
        self.service_values['redis'].config(
            text=redis_text, fg='#22c55e' if snapshot.redis_online else '#ef4444'
        )
        self.service_values['alphabet_workers'].config(
            text=str(snapshot.alphabet_workers_online)
        )
        self.service_values['phrase_workers'].config(
            text=str(snapshot.phrase_workers_online)
        )
        if snapshot.last_request_id:
            self.last_request_id = snapshot.last_request_id
        if snapshot.last_worker_id:
            self.last_worker_id = snapshot.last_worker_id
        if snapshot.last_processing_ms is not None:
            self.last_processing_ms = snapshot.last_processing_ms
        if snapshot.last_task_status:
            self.last_task_status = snapshot.last_task_status
        self._render_last_task_fields()
        if not snapshot.coordinator_online:
            self.status.config(text='● Coordinator offline', fg='#ef4444')
        elif not snapshot.redis_online:
            self.status.config(text='● API online / Redis offline', fg='#fbbf24')
        else:
            self.status.config(text='● Distributed services online', fg='#22c55e')

    def close(self):
        self.closing = True
        self.status_poller.stop()
        self.running = False
        try: self.camera.stop()
        except Exception: pass
        self.root.destroy()

def main() -> None:
    root = tk.Tk()
    app = ASLApp(root)
    root.protocol('WM_DELETE_WINDOW', app.close)
    root.mainloop()

if __name__ == '__main__':
    main()
