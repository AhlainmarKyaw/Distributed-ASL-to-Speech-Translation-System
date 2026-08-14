@echo off
call .venv\Scripts\activate
celery -A workers.celery_app.celery_app worker -Q phrase_queue -P solo -n phrase@%%h --loglevel=info
