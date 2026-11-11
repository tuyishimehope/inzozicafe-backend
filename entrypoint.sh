#!/bin/sh

set -e

echo "Running migrations..."

alembic upgrade head

echo "Starting API..."

# Extra arguments are passed to uvicorn: docker-compose adds --reload for development.
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 "$@"
