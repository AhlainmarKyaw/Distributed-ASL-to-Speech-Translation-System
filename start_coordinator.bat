@echo off
call .venv\Scripts\activate
if not defined API_HOST set API_HOST=127.0.0.1
if not defined API_PORT set API_PORT=8000
python -m uvicorn coordinator.main:app --host %API_HOST% --port %API_PORT%
