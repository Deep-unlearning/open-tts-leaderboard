FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    HF_HOME=/tmp/hf_home \
    HF_HUB_CACHE=/hf_cache/hub \
    HF_DATASETS_CACHE=/hf_cache/datasets_tts_api \
    NUMBA_CACHE_DIR=/tmp/numba_cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY api/requirements.txt /app/api/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/api/requirements.txt

COPY api/*.py /app/api/
COPY api/providers /app/api/providers
COPY scripts/run_eval_utils.py scripts/ttfa_probe.py /app/scripts/

WORKDIR /app/api
CMD ["python", "run_eval.py", "--list_models"]
