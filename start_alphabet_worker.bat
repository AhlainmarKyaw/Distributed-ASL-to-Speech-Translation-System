@echo off
call .venv\Scripts\activate
if not defined ALPHABET_QUEUE set ALPHABET_QUEUE=alphabet_queue
celery -A workers.celery_app.celery_app worker -Q %ALPHABET_QUEUE% -P solo -n alphabet@%%h --loglevel=info
