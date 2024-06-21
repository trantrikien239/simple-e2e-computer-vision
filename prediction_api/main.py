import os
import base64
import io
import time
from fastapi import FastAPI, HTTPException, Request
from PIL import Image
import torch

from .utils import ImageData
from .utils import get_device, load_model, get_image_transform
from .utils import decode_image, classify_image

app = FastAPI()

device = get_device()

model_path = os.environ.get("MODEL_PATH", "model.pth")
model = load_model(model_path, device)
image_transform = get_image_transform()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.post("/predict/")
async def predict(data: ImageData):
    try:
        image = decode_image(data.image)
        prediction = await classify_image(image, model, image_transform, device)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))