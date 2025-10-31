FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Proxy & timeout settings for build stage (optional)
ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG UV_HTTP_TIMEOUT=120

ENV http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy} \
    UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT}

LABEL org.opencontainers.image.description="Web-based Gemini models wrapped into an OpenAI-compatible API."

WORKDIR /app

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --no-cache --no-dev

COPY app/ app/
COPY config/ config/
COPY run.py .

# Runtime proxy & timeout can be overridden by docker run/compose environment
ENV UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT}

# Command to run the application
CMD ["uv", "run", "--no-dev", "run.py"]
