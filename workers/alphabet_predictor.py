from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("ALPHABET_MODEL", BASE / "models" / "alphabet_landmarks.keras"))
LABEL_PATH = BASE / "models" / "alphabet_labels.json"

class AlphabetPredictor:
    def __init__(self, confidence_threshold: float = 0.70):
        self.threshold = confidence_threshold
        self.model = None
        self.labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self._load()

    def _load(self):
        if LABEL_PATH.exists():
            self.labels = json.loads(LABEL_PATH.read_text(encoding="utf-8"))
        if MODEL_PATH.exists():
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
            from tensorflow import keras
            self.model = keras.models.load_model(MODEL_PATH, compile=False)

    @property
    def ready(self): return self.model is not None

    def predict(self, features):
        if self.model is None:
            return {"status":"model_missing","letter":None,"confidence":None,"inference_ms":None,
                    "message":"Train the alphabet model first: python training/train_alphabet.py"}
        x = np.asarray(features, dtype=np.float32)
        if x.shape != (63,):
            raise ValueError("Expected 63 MediaPipe landmark features")
        t = time.perf_counter()
        probs = np.asarray(self.model.predict(x[None, :], verbose=0))[0]
        ms = (time.perf_counter()-t)*1000
        i = int(np.argmax(probs)); conf = float(probs[i])
        return {"status":"predicted" if conf >= self.threshold else "uncertain",
                "letter":self.labels[i] if conf >= self.threshold else None,
                "confidence":conf,"inference_ms":ms}
