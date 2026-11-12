# ml-nlp-eks-proto/services/alerting/Dockerfile
FROM python:3.13.12-slim
WORKDIR /app

# Копіюємо requirements сервісу (шлях від кореня)
COPY services/ml_infer/requirements.txt services/ml_infer/constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Копіюємо спільний модуль і код сервісу (шляхи від кореня)
COPY common /app/common
COPY services/ml_infer/main.py .
COPY models/ml/ml_config.json .

ENV PORT=8000
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]