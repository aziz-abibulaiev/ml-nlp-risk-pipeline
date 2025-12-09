FROM python:3.13-slim
WORKDIR /app

COPY services/fusion/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common /app/common
COPY services/fusion/main.py .

ENV PORT=8000
CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "--timeout", "30"]