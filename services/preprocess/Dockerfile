FROM python:3.13-slim
WORKDIR /app

COPY services/preprocess/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common /app/common
COPY services/preprocess/main.py .

ENV PORT=8000
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]