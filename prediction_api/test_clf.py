import os
import sys
import asyncio
import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from main import app_state, initialize_app_state
from utils import classify_image

app_state = asyncio.run(initialize_app_state())

def test_model_loaded():
    assert app_state.model is not None

def test_device_loaded():
    assert app_state.device is not None

def test_image_transform_loaded():
    assert app_state.image_transform is not None

@pytest.mark.parametrize(
    "test_set, accuracy", [
        ("../data/test-sample-1000-seed-42/normal", 0.98),
        ("../data/test-sample-1000-seed-42/normal", 0.95),
        ("../data/test-sample-1000-seed-42/normal", 0.90),
        ("../data/test-sample-1000-seed-42/noisy", 0.98),
        ("../data/test-sample-1000-seed-42/noisy", 0.95),
        ("../data/test-sample-1000-seed-42/noisy", 0.90),
        ("../data/test-sample-1000-seed-42/bright", 0.98),
        ("../data/test-sample-1000-seed-42/bright", 0.95),
        ("../data/test-sample-1000-seed-42/bright", 0.90),
    ]
)
def test_acurracy_normal(test_set, accuracy):
    # Load the test data from the directory
    test_files = os.listdir(test_set)
    y_pred = []
    for test_file in test_files:
        image = Image.open(os.path.join(test_set, test_file)).convert('L')
        prediction = classify_image(
            image,
            app_state.model,
            app_state.image_transform,
            app_state.device
        )
        y_pred.append(prediction)
    y_true = [int(test_file.split('_')[0]) for test_file in test_files]
    test_acc = sum([1 for yp, yt in zip(y_pred, y_true) if yp == yt]) / len(y_true)

    assert test_acc >= accuracy