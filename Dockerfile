# Two targets share one base:
#   docker build -t loyalty-spend .                     analysis CLI (default)
#   docker build --target serve -t loyalty-spend-api .  scoring service
FROM python:3.12-slim AS base

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

FROM base AS serve

RUN pip install ".[serve]" \
    && useradd --create-home --uid 10001 app
USER app
ENV LOYALTY_MODEL=/app/artifacts/model.joblib
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
# Mount a trained bundle: -v $(pwd)/artifacts:/app/artifacts
CMD ["loyalty-spend", "serve", "--host", "0.0.0.0", "--port", "8000", "--model", "/app/artifacts/model.joblib"]

FROM base AS cli

COPY data ./data
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install ".[gbm,deep]"

ENTRYPOINT ["loyalty-spend"]
CMD ["tasks"]
