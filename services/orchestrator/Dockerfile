FROM python:3.13-slim
WORKDIR /app

COPY services/orchestrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common /app/common
COPY services/orchestrator/main.py services/orchestrator/policy.py ./

ENV PORT=8000
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]