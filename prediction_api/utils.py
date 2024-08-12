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
    """
    Load the model from the specified path and move it to the specified device.

    Args:
        model_path (str): The path to the model file
        device (torch.device): The device to move the model to
    Returns:
        torch.nn.Module: The loaded model
    """
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
    """
    Decode the base64-encoded image data and convert it to a PIL image.
    
    Args:
        image_base64 (str): The base64-encoded image data
    Returns:
        PIL.Image: The decoded image
    """
    image_data = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_data)).convert('L')

def classify_image(image, model, transform, device):
    """
    Classify the image using the model.
    
    Args:
        image (PIL.Image): The input image
        model (torch.nn.Module): The model to use for classification
        transform (torchvision.transforms.Compose): The image transform
        device (torch.device): The device to use for computation
    Returns:
        int: The predicted class
    """
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    prediction_result = output.argmax(dim=1, keepdim=True).item()
    return prediction_result