@echo off
if not exist .venv py -3.12 -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt
echo Setup complete.
