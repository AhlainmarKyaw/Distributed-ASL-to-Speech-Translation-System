from __future__ import annotations
from pydantic import BaseModel, Field

class AlphabetRequest(BaseModel):
    features: list[float] = Field(min_length=63, max_length=63)

class AlphabetResponse(BaseModel):
    status: str
    letter: str | None = None
    confidence: float | None = None
    inference_ms: float | None = None
    buffer_size: int

class SystemStatus(BaseModel):
    status: str
    redis: bool
    alphabet_worker: bool
    phrase_worker: bool
