FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

COPY . /app

WORKDIR /app

RUN uv sync --frozen --no-dev

FROM python:3.13-slim-bookworm

COPY --from=builder /app/.venv/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /app/src /app

WORKDIR /app

CMD ["python", "main.py"]
