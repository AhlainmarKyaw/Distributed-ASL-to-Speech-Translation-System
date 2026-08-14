from __future__ import annotations
import os
from fastapi import FastAPI, HTTPException
from redis import Redis
from pydantic import BaseModel, Field
from workers.alphabet_tasks import predict_alphabet
from workers.phrase_tasks import predict_phrase, normalize_phrase

app=FastAPI(title="Distributed ASL-to-Speech Translation System",version="2.0.0",
 description="Distributed ASL recognition using FastAPI, Redis and specialized Celery AI workers.")

class AlphabetRequest(BaseModel):
    features:list[float]=Field(min_length=63,max_length=63)
class PhraseRequest(BaseModel):
    sequence:list[list[float]]

@app.get("/")
def root(): return {"project":"Distributed ASL-to-Speech Translation System","docs":"/docs","health":"/health"}

@app.get("/health")
def health():
    ok=False
    try:
        Redis.from_url(os.getenv("REDIS_URL","redis://127.0.0.1:6379/0")).ping(); ok=True
    except Exception: pass
    return {"status":"ok","redis":ok,"architecture":"Client → FastAPI → Redis → Celery workers → TensorFlow → Speech"}

@app.post("/predict/alphabet")
def alphabet(req:AlphabetRequest):
    try:
        return predict_alphabet.apply_async(args=[req.features],queue="alphabet_queue").get(timeout=10,disable_sync_subtasks=False)
    except Exception as e: raise HTTPException(503,f"Alphabet worker unavailable: {e}")

@app.post("/predict/phrase")
def phrase_predict(req:PhraseRequest):
    if len(req.sequence)!=30 or any(len(x)!=63 for x in req.sequence):
        raise HTTPException(400,"sequence must be 30 frames × 63 features")
    try:
        return predict_phrase.apply_async(args=[req.sequence],queue="phrase_queue").get(timeout=12,disable_sync_subtasks=False)
    except Exception as e: raise HTTPException(503,f"Phrase worker unavailable: {e}")

@app.post("/phrase")
def phrase(payload:dict):
    text=str(payload.get("text","")).strip()
    if not text: raise HTTPException(400,"text is required")
    try:
        return normalize_phrase.apply_async(args=[text],queue="phrase_queue").get(timeout=6,disable_sync_subtasks=False)
    except Exception as e: raise HTTPException(503,f"Phrase worker unavailable: {e}")
