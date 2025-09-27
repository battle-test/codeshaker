FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt ./
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY services/rerank ./services/rerank

ENV PYTHONPATH=/app/src

EXPOSE 8081
CMD ["uvicorn", "services.rerank.main:app", "--host", "0.0.0.0", "--port", "8081"]
