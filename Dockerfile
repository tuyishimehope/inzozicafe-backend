FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install \
    --default-timeout=300 \
    --no-cache-dir \
    -r requirements.txt

COPY . .

RUN chmod +x ./entrypoint.sh

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
