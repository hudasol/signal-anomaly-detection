# Signal: reproducible images for the inference service and the test suite.
#
#   docker build -t signal .                                    # runtime image (API)
#   docker run --rm -p 8000:8000 signal                         # serves the shipped model
#   docker build --target test -t signal-test .                 # runtime + test tools + tests
#   docker run --rm signal-test                                 # pytest
#   docker run --rm signal-test sh -c "signal-data generate && signal-eval show"
#
# Base image pinned by digest (python:3.12.3-slim, the interpreter the results were
# produced with). Every package is pinned by version and hash.
ARG BASE=python:3.12.3-slim@sha256:afc139a0a640942491ec481ad8dda10f2c5b753f5c969393b12480155fe15a63

# ---------------------------------------------------------------- build: a venv with the package
FROM ${BASE} AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN python -m venv /venv
COPY requirements.runtime.lock /tmp/
RUN /venv/bin/pip install --require-hashes -r /tmp/requirements.runtime.lock
COPY pyproject.toml README.md /src/
COPY src /src/src
RUN /venv/bin/pip install --no-deps /src

# ---------------------------------------------------------------- runtime: API only, non-root
FROM ${BASE} AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH=/venv/bin:$PATH SIGNAL_HOME=/app
RUN useradd --create-home --uid 10001 signal
WORKDIR /app
COPY --from=build /venv /venv
# Only what serving and the read-only demo commands need: configs, the evaluated
# artifacts + registry, and the saved results the demo commands read.
COPY configs ./configs
COPY models ./models
COPY results/validation ./results/validation
COPY results/official ./results/official
COPY results/exceeds/drift_reference.json ./results/exceeds/drift_reference.json
RUN mkdir -p data results/replay results/demo && chown -R signal:signal data results
USER signal
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/livez', timeout=2).status != 200)"
CMD ["uvicorn", "fleet_signal.service.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------- test: runtime + dev tools + tests
FROM runtime AS test
USER root
COPY requirements.lock /tmp/
RUN /venv/bin/pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock
COPY pyproject.toml ./
COPY tests ./tests
COPY src ./src
COPY docs ./docs
RUN chown -R signal:signal /app
USER signal
CMD ["pytest"]

# ---------------------------------------------------------------- default target: the runtime image
FROM runtime AS serve
