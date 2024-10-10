FROM --platform=linux/arm64 python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_PATH=/app/model_registry/latest/scripted_model.pt
# ENV MODEL_PATH=/app/model_registry/production/scripted_model.pt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--port", "8000", "--host", "0.0.0.0"]