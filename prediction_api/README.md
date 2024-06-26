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

## Usage

1. Build the Docker image:
```bash
docker build -t prediction-api .
```

2. Run the Docker container:

```bash
docker run -p 80:80 prediction-api
```

## API Endpoints
- `/predict`: POST endpoint that accepts input data and returns predictions from the pre-trained model.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details