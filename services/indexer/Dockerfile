FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt ./
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY services/indexer ./services/indexer

ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uvicorn", "services.indexer.main:app", "--host", "0.0.0.0", "--port", "8000"]
