from __future__ import annotations
from functools import lru_cache
from workers.celery_app import celery_app
from workers.alphabet_predictor import AlphabetPredictor

@lru_cache(maxsize=1)
def _predictor(): return AlphabetPredictor()

@celery_app.task(name="workers.alphabet_tasks.predict_alphabet")
def predict_alphabet(features: list[float]) -> dict:
    return _predictor().predict(features)
