from pydantic import BaseModel

from PIL import Image
import base64
import io

import torch
from torchvision import transforms


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

async def load_model(model_path, device):
    model = torch.jit.load(model_path)
    model = model.to(device)
    model.eval()
    return model

async def get_image_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


class ImageData(BaseModel):
    image: str

class ImageDataBatch(BaseModel):
    images: list


def decode_image(image_base64):
    image_data = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_data)).convert('L')

def classify_image(image, model, transform, device):
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    prediction_result = output.argmax(dim=1, keepdim=True).item()
    return prediction_result