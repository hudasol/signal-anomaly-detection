# Signal: reproducible environment for the pipeline and the inference service.
#   docker build -t signal .
#   docker run --rm signal pytest                                  # tests
#   docker run --rm -p 8000:8000 signal                            # API (serves the shipped model)
#   docker run --rm signal sh -c "signal-data generate && signal-eval show"
FROM python:3.12.3-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .

COPY configs ./configs
COPY tests ./tests
COPY models ./models
COPY results ./results
COPY docs ./docs

EXPOSE 8000
CMD ["uvicorn", "fleet_signal.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
