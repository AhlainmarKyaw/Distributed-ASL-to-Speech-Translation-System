@echo off
call .venv\Scripts\activate
if not defined PHRASE_QUEUE set PHRASE_QUEUE=phrase_queue
celery -A workers.celery_app.celery_app worker -Q %PHRASE_QUEUE% -P solo -n phrase@%%h --loglevel=info
