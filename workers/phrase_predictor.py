from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("PHRASE_MODEL", BASE / "models" / "phrase_sequence.keras"))
LABEL_PATH = BASE / "models" / "phrase_labels.json"

class PhrasePredictor:
    def __init__(self, threshold=0.72):
        self.threshold=threshold; self.model=None; self.labels=[]
        if LABEL_PATH.exists(): self.labels=json.loads(LABEL_PATH.read_text(encoding="utf-8"))
        if MODEL_PATH.exists():
            os.environ.setdefault("CUDA_VISIBLE_DEVICES","-1")
            from tensorflow import keras
            self.model=keras.models.load_model(MODEL_PATH, compile=False)

    @property
    def ready(self): return self.model is not None

    def predict(self, sequence):
        if self.model is None:
            return {"status":"model_missing","phrase":None,"confidence":None,
                    "message":"Train phrase model first: python training/train_phrases.py"}
        x=np.asarray(sequence,dtype=np.float32)
        if x.shape != (30,63): raise ValueError("Expected phrase sequence shape (30, 63)")
        t=time.perf_counter(); p=np.asarray(self.model.predict(x[None,...],verbose=0))[0]
        ms=(time.perf_counter()-t)*1000; i=int(np.argmax(p)); c=float(p[i])
        return {"status":"predicted" if c>=self.threshold else "uncertain",
                "phrase":self.labels[i] if c>=self.threshold else None,
                "confidence":c,"inference_ms":ms}
