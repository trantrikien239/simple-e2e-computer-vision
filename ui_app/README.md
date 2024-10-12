# Streamlit Image Classification App

This Streamlit app allows users to upload a PNG image, calls a POST API to classify the image, and displays both the image and the classification result.

## Features

- Upload a PNG image
- Call a POST API to get the classification of the image
- Display the uploaded image and the prediction

## How to run

```bash
streamlit run app.py
```

## Build with Docker

The UI is dependent on the prediction api. For this reason, see ../README.md for multi-container application deployment.

## Heroku deployment

### Set up - run these only once

```bash
heroku login
# Set up the UI app
heroku create e2e-model-development
heroku stack:set container -a e2e-model-development
# Point the UI to the prediction API - change the link accordingly
heroku config:set API_URL=https://link-to-prediction-api.herokuapp.com -a e2e-model-development
```

### Deploy - run these every time there's a new version of the app

```bash
# Deploy the UI app
cd ../ui_app
heroku container:push web -a e2e-model-development
heroku container:release web -a e2e-model-development
```