FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN useradd --create-home edgevision
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY edgevision ./edgevision
COPY configs ./configs
RUN mkdir -p /app/data && chown -R edgevision:edgevision /app/data
USER edgevision
EXPOSE 8080
CMD ["python", "-m", "edgevision.cli", "serve", "--host", "0.0.0.0", "--port", "8080", "--source", "synthetic", "--db", "/app/data/events.sqlite3"]
