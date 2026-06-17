FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt || true

COPY . /app

EXPOSE 8008
ENV URCM_METRICS_BIND=0.0.0.0

CMD ["python", "-m", "urcm.ops.metrics_exporter"]
