from __future__ import annotations
import re
from functools import lru_cache
from workers.celery_app import celery_app
from workers.phrase_predictor import PhrasePredictor

COMMON = {
 "HELLO":"Hello","HI":"Hi","THANKYOU":"Thank you","THANKS":"Thank you",
 "GOODMORNING":"Good morning","PLEASE":"Please","YES":"Yes","NO":"No",
 "HELP":"Help me","HELPME":"Help me","SORRY":"Sorry","ILOVEYOU":"I love you",
 "HOWAREYOU":"How are you?","NICETOMEETYOU":"Nice to meet you","GOODBYE":"Goodbye","BYE":"Goodbye"
}
@lru_cache(maxsize=1)
def _predictor(): return PhrasePredictor()

@celery_app.task(name="workers.phrase_tasks.predict_phrase")
def predict_phrase(sequence: list[list[float]]) -> dict:
    return _predictor().predict(sequence)

@celery_app.task(name="workers.phrase_tasks.normalize_phrase")
def normalize_phrase(text: str) -> dict:
    key=re.sub(r"[^A-Za-z]","",text).upper()
    out=COMMON.get(key,text.strip())
    return {"input":text,"text":out,"matched":out != text.strip()}
