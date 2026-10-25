# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos

# quantoniumos quick validation container
FROM python:3.11-slim AS builder

# Security patch: Purge phantom kernel files and update system packages
RUN apt-get update && apt-get upgrade -y tar nasm && apt-get remove --purge -y '^linux-image-.*' '^linux-headers-.*' && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*


ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=offscreen

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential nasm git ca-certificates && \
    apt-get upgrade -y && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /qos

# Copy only what we need to build/install first (better caching).
COPY pyproject.toml README.md LICENSE* NOTICE* ./
COPY src/ ./src/
COPY algorithms/ ./algorithms/
COPY quantonium_os_src/ ./quantonium_os_src/
COPY tools/ ./tools/

RUN python -m pip install --upgrade pip && \
    python -m pip wheel --wheel-dir=/wheels . && \
    python -m pip wheel --wheel-dir=/wheels pytest hypothesis

# Optional native kernels (best-effort; tests still run without them).
RUN make -C algorithms/rft/kernels all || true

FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=offscreen

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && \
    apt-get upgrade -y && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /qos

COPY --from=builder /wheels /wheels
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Copy the repo for pytest collection (without bundling build tools into the image).
COPY . /qos

# Copy compiled native outputs if they exist.
COPY --from=builder /qos/algorithms/rft/kernels/compiled/ /qos/algorithms/rft/kernels/compiled/

CMD ["pytest", "-m", "not slow"]
