# Prediction API

This project contains a prediction API built with FastAPI and deployed using Docker.

## Overview

The API is designed to make predictions using a pre-trained model. FastAPI is used for setting up the API due to its high performance and easy-to-use features. Docker is used for deployment to ensure the application runs seamlessly in any environment.

## File Structure
```
. 
├── main.py 
├── Dockerfile 
└── README.md
```

- `main.py`: This is the main FastAPI application file where the API endpoints are defined.
- `Dockerfile`: This file is used by Docker to build a Docker image of the application.

## Run the server

```bash
export MODEL_PATH=../model_registry/latest/scripted_model.pt
uvicorn prediction_api.main:app --reload
```

## Build with Docker

1. Build the Docker image:
```bash
cd ..
./copy_files.sh
cd prediction_api
docker build -f Dockerfile -t prediction-api .
```

2. Run the Docker container:

```bash
docker run -p 8000:8000 prediction-api
```

## API Endpoints
- `/predict`: POST endpoint that accepts input data and returns prediction from the pre-trained model.
- `/predict_batch`: POST endpoint that accepts a list of input images and return a list of predictions.
