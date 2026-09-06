FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr libtesseract-dev libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV TESSERACT_CMD=/usr/bin/tesseract
ENV PYTHONUNBUFFERED=1

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
