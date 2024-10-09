FROM --platform=linux/arm64 python:3.11-slim

WORKDIR /user/src/app

COPY ../requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY ./prediction_api .
COPY ../model_registry/latest/scripted_model.pt ./

ENV MODEL_PATH=scripted_model.pt

CMD ["uvicorn", "main:app", "--port", "8000", "--host", "0.0.0.0"]