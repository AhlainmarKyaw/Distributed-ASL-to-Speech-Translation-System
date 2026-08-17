FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_VISIBLE_DEVICES=-1

WORKDIR /app

RUN groupadd --system asl && useradd --system --gid asl --create-home asl

COPY requirements-server.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements-server.txt

COPY --chown=asl:asl config.py observability.py input_validation.py ./
COPY --chown=asl:asl coordinator ./coordinator
COPY --chown=asl:asl workers ./workers
COPY --chown=asl:asl scripts/container_healthcheck.py ./scripts/container_healthcheck.py
COPY --chown=asl:asl models ./models

USER asl

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "coordinator.main:app", "--host", "0.0.0.0", "--port", "8000"]
