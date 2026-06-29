# Stage 1: Build
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    URCM_METRICS_BIND=0.0.0.0 \
    URCM_METRICS_PORT=8008

WORKDIR /app

COPY --from=builder /install /usr/local
COPY urcm/ /app/urcm/

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request, json, sys; r = urllib.request.urlopen('http://localhost:8008/health', timeout=5); d = json.loads(r.read()); sys.exit(0 if d.get('ok') else 1)"

EXPOSE 8008

CMD ["python", "-m", "uvicorn", "urcm.api:app", "--host", "0.0.0.0", "--port", "8008"]
