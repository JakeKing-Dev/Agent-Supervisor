# Distroless-style: no package installs, because the library has no
# dependencies. Small image, small attack surface, fast cold start.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Source only; nothing to pip install.
COPY supervisor/ /app/supervisor/
COPY examples/ /app/examples/

# Run unprivileged.
RUN useradd --create-home --uid 10001 supervisor \
 && chown -R supervisor:supervisor /app
USER 10001

EXPOSE 9464

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
  CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:9464/healthz',timeout=2).status==200 else 1)"

CMD ["python3", "examples/service.py"]
