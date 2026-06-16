FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip --no-cache-dir \
 && pip install --no-cache-dir -r requirements.txt

COPY src/       ./src/
COPY main.py    ./main.py
COPY static/    ./static/

RUN mkdir -p logs models data/raw data/processed

# Train model at build time — baked into image
RUN python -m src.components.model_trainer

ENV TF_CPP_MIN_LOG_LEVEL=3 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
