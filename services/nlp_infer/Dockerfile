FROM python:3.13.12-slim
WORKDIR /app

COPY services/nlp_infer/requirements.txt services/nlp_infer/constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY common /app/common
COPY services/nlp_infer/main.py .

ENV PORT=8000
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]