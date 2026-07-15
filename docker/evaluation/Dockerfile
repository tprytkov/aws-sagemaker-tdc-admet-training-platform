FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/program/src

WORKDIR /opt/program

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY sagemaker/evaluation_requirements.txt ./evaluation_requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r evaluation_requirements.txt

COPY sagemaker/evaluate_models.py ./evaluate_models.py
COPY src/admet_platform ./src/admet_platform
COPY configs ./configs

ENTRYPOINT ["python", "/opt/program/evaluate_models.py"]
