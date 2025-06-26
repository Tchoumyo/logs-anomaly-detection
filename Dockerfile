FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock /app/

RUN poetry install --no-root

COPY models /app/models

COPY src /app/src

COPY logs /app/logs

EXPOSE 8000

CMD [ "poetry", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000" ]