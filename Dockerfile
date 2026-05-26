FROM python:3.11-slim

WORKDIR /app

# Системные зависимости для LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY tests/ ./tests/
COPY app/    ./app/

CMD ["python", "-m", "pytest", "tests/", "-v"]
