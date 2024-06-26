import os
import time
from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
import asyncio

from .utils import ImageData
from .utils import get_device, load_model, get_image_transform
from .utils import decode_image, classify_image

class AppState:
    def __init__(self):
        self.model_path = None
        self.device = None
        self.model = None
        self.image_transform = None

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start up
    app_state.model_path = os.environ.get("MODEL_PATH", "model.pth")
    app_state.device = get_device()

    # Gather asynchronous tasks
    model_task = asyncio.create_task(
        load_model(
            app_state.model_path, 
            app_state.device
            )
        )
    transform_task = asyncio.create_task(get_image_transform())
    # Wait for the tasks to complete
    app_state.model, app_state.image_transform = \
        await asyncio.gather(model_task, transform_task)
    
    yield
    # Shut down
    del app_state.model_path
    del app_state.device
    del app_state.model
    del app_state.image_transform

app = FastAPI(lifespan=lifespan)

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
        prediction = classify_image(
            image,
            app_state.model,
            app_state.image_transform,
            app_state.device
            )
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))