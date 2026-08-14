from __future__ import annotations
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox

import cv2
import requests
from PIL import Image, ImageTk

from client.alphabet_buffer import AlphabetBuffer
from client.camera import Camera
from client.voice_service import VoiceService

API = 'http://127.0.0.1:8000'
COMMON_PHRASES = ['Hello', 'Thank you', 'Good morning', 'Please', 'Yes', 'No', 'Help me', 'Sorry', 'I love you', 'How are you?', 'Nice to meet you', 'Goodbye']

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
        self._build_ui()
        self._check_backend()

    def _build_ui(self) -> None:
        top = tk.Frame(self.root, bg='#08111f', padx=28, pady=22)
        top.pack(fill='x')
        tk.Label(top, text='Distributed ASL-to-Speech', bg='#08111f', fg='#6ee7ff', font=('Segoe UI', 27, 'bold')).pack(side='left')
        tk.Label(top, text='Translation System', bg='#08111f', fg='#d7e4f5', font=('Segoe UI', 13)).pack(side='left', padx=16, pady=(10,0))
        self.status = tk.Label(top, text='● checking services', bg='#08111f', fg='#fbbf24', font=('Segoe UI', 11, 'bold'))
        self.status.pack(side='right')

        main = tk.Frame(self.root, bg='#08111f', padx=28)
        main.pack(fill='both', expand=True)
        left = tk.Frame(main, bg='#0d1a2c', padx=16, pady=16, highlightbackground='#1d3655', highlightthickness=1)
        left.pack(side='left', fill='both', expand=True, padx=(0, 14))
        right = tk.Frame(main, bg='#0d1a2c', width=390, padx=22, pady=20, highlightbackground='#1d3655', highlightthickness=1)
        right.pack(side='right', fill='y')
        right.pack_propagate(False)

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
        self.current = tk.Label(right, text='—', bg='#0d1a2c', fg='#6ee7ff', font=('Segoe UI', 50, 'bold'))
        self.current.pack(anchor='w', pady=(6, 10))
        tk.Label(right, text='Sentence / fingerspelling', bg='#0d1a2c', fg='#d7e4f5', font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        self.text = tk.Text(right, height=5, wrap='word', bg='#07101d', fg='white', insertbackground='white', relief='flat', font=('Segoe UI', 16), padx=12, pady=10)
        self.text.pack(fill='x', pady=(8, 10))
        actions = tk.Frame(right, bg='#0d1a2c')
        actions.pack(fill='x')
        self._button(actions, '+ Space', self.add_space, '#334155').pack(side='left')
        self._button(actions, '⌫ Delete', self.delete, '#334155').pack(side='left', padx=8)
        self._button(actions, 'Smart Phrase', self.smart_phrase, '#8b5cf6').pack(side='left')

        tk.Label(right, text='COMMON PHRASES', bg='#0d1a2c', fg='#7f93ad', font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(22, 8))
        chips = tk.Frame(right, bg='#0d1a2c')
        chips.pack(fill='x')
        for phrase in COMMON_PHRASES:
            b = tk.Button(chips, text=phrase, command=lambda p=phrase:self.use_phrase(p), bg='#14253d', fg='#d7e4f5', activebackground='#1d3655', activeforeground='white', relief='flat', cursor='hand2', font=('Segoe UI', 9), padx=8, pady=5)
            b.pack(side='left', padx=(0,6), pady=4)
            if sum(w.winfo_reqwidth() for w in chips.winfo_children()) > 340:
                # Tk does not wrap frames; keep compact by allowing natural clipping on tiny screens.
                pass
        tk.Label(right, text='Architecture: Webcam → MediaPipe → FastAPI → Redis → specialized Celery AI workers → text → speech', bg='#0d1a2c', fg='#617691', wraplength=340, justify='left', font=('Segoe UI', 9)).pack(side='bottom', anchor='w', pady=8)

    def _button(self, parent, text, command, color):
        return tk.Button(parent, text=text, command=command, bg=color, fg='white', activebackground=color, activeforeground='white', relief='flat', cursor='hand2', font=('Segoe UI', 10, 'bold'), padx=14, pady=8)

    def _check_backend(self):
        def work():
            try:
                data = requests.get(API + '/health', timeout=2).json()
                ok = data.get('redis', False)
                self.root.after(0, lambda:self.status.config(text='● Distributed services online' if ok else '● API online / Redis offline', fg='#22c55e' if ok else '#fbbf24'))
            except Exception:
                self.root.after(0, lambda:self.status.config(text='● Coordinator offline', fg='#ef4444'))
        threading.Thread(target=work, daemon=True).start()

    def toggle_camera(self):
        if self.running:
            self.running = False
            self.camera.stop()
            self.video.config(image='', text='Camera stopped')
            self.start_btn.config(text='▶ Start Camera', bg='#22c55e')
            return
        try:
            self.camera = Camera(0)
            self.camera.start()
            self.running = True
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
            w, h = max(640, self.video.winfo_width()), max(420, self.video.winfo_height())
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
        try:
            r = requests.post(API + '/predict/alphabet', json={'features': features}, timeout=10)
            r.raise_for_status()
            self.result_queue.put(('ok', r.json()))
        except Exception as exc:
            self.result_queue.put(('err', str(exc)))
        finally:
            self.inflight = False

    def _drain_results(self):
        try:
            while True:
                kind, payload = self.result_queue.get_nowait()
                if kind == 'err':
                    self.status.config(text='● worker unavailable', fg='#ef4444')
                    continue
                if payload.get('status') == 'model_missing':
                    self.status.config(text='● Train alphabet model first', fg='#fbbf24')
                letter = payload.get('letter')
                confidence = payload.get('confidence')
                self.conf.config(text=f"Confidence: {confidence:.1%}" if confidence is not None else 'Confidence: —')
                self.current.config(text=letter or '…')
                if letter:
                    self._stabilize(letter)
        except queue.Empty:
            pass

    def _stabilize(self, letter):
        if letter == self.last_letter:
            self.streak += 1
        else:
            self.last_letter, self.streak = letter, 1
        # worker model already consumes a 30-frame sequence; UI adds a second light debounce.
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
        if not raw: return
        def work():
            try:
                r = requests.post(API + '/phrase', json={'text': raw}, timeout=6)
                r.raise_for_status(); normalized = r.json()['text']
                self.root.after(0, lambda:self._apply_phrase(normalized))
            except Exception as exc:
                self.root.after(0, lambda:messagebox.showwarning('Phrase worker', str(exc)))
        threading.Thread(target=work, daemon=True).start()
    def _apply_phrase(self, text):
        self.buffer.set_text(text); self._sync_text()

    def close(self):
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
