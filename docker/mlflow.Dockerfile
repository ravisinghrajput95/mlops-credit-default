# MLflow tracking server with a Postgres driver baked in. Installing the driver
# at container start instead would add latency to every boot and make startup
# depend on PyPI being reachable.
FROM ghcr.io/mlflow/mlflow:v3.15.2

RUN pip install --no-cache-dir psycopg2-binary==2.9.10

EXPOSE 5000
