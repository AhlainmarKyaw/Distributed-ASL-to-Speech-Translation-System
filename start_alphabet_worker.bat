@echo off
call .venv\Scripts\activate
celery -A workers.celery_app.celery_app worker -Q alphabet_queue -P solo -n alphabet@%%h --loglevel=info
