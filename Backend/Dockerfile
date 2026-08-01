# Cloud Run injects $PORT at runtime (default 8080) and the container
# must bind to it — shell form CMD (not exec/JSON-array form) so
# ${PORT:-8080} actually gets expanded by a shell before uvicorn runs;
# exec form passes args straight to execve with no shell involved, so
# "$PORT" would reach uvicorn as a literal, unexpanded string.
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
