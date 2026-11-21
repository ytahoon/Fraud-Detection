FROM python:3.13.5-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --timeout 60 --resume-retries 5 -r requirements.txt && python -m pip check

COPY src ./src
COPY artifacts/models ./artifacts/models
COPY artifacts/dashboard/*.json ./artifacts/dashboard/
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY tests ./tests
