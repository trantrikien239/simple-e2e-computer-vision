import os
import base64
import io
import time
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from PIL import Image
import torch
from torchvision import transforms

app = FastAPI()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

class ImageData(BaseModel):
    image: str

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def load_model(model_path, device):
    model = torch.jit.load(model_path)
    model = model.to(device)
    model.eval()
    return model

def get_image_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

device = get_device()
print(f"Using device: {device}")

model_path = os.environ.get("MODEL_PATH", "model.pth")
print(f"Loading model from: {model_path}")
model = load_model(model_path, device)
image_transform = get_image_transform()

@app.post("/predict/")
async def predict(data: ImageData):
    try:
        image = decode_image(data.image)
        prediction = await classify_image(image, model, image_transform)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def decode_image(image_base64):
    image_data = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_data)).convert('L')

async def classify_image(image, model, transform):
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    prediction_result = output.argmax(dim=1, keepdim=True).item()
    return prediction_result
