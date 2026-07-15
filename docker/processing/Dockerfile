FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/program/src

WORKDIR /opt/program

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY sagemaker/processing_requirements.txt ./processing_requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r processing_requirements.txt

COPY sagemaker/prepare_tdc_dataset.py ./prepare_tdc_dataset.py
COPY src/admet_platform ./src/admet_platform
COPY configs ./configs

ENTRYPOINT ["python", "/opt/program/prepare_tdc_dataset.py"]
