@echo off
call .venv\Scripts\activate
python -m uvicorn coordinator.main:app --host 127.0.0.1 --port 8000
