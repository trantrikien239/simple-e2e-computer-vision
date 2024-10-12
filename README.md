## Installing the environment

```bash
pip install -r requirements.txt
```

## Train

```bash
python model/train.py
```

## Deploy with Docker
```
./copy_files.sh
docker-compose up --build
```

This will build and start the Prediction API and the UI as separate services that are connected via a virtual network. Access locally:
    - Prediction API docs: `http://localhost:8000/docs`
    - Streamlit UI: `http://localhost:8501/`

![Prediction API - Backend](_assets/fastapi_backend.png)

![Streamlit Frontend](_assets/streamlit_frontend.png)

## Heroku deployment

Heroku support container deployment, however, multi-container apps are not supported afaik. So, the prediction API and the UI app will need to be deployed separately.

### Set up - run these only once

```bash
heroku login
# Set up the prediction API
heroku create mnist-prediction-api
heroku stack:set container -a mnist-prediction-api
# Set up the UI app
heroku create e2e-model-development
heroku stack:set container -a e2e-model-development
# Point the UI to the prediction API - change the link accordingly
heroku config:set API_URL=https://link-to-prediction-api.herokuapp.com -a e2e-model-development
```

### Deploy - run these every time there's a new version of the app

```bash
# Deploy the prediction API
cd prediction_api
heroku container:push web -a mnist-prediction-api
heroku container:release web -a mnist-prediction-api
# Deploy the UI app
cd ../ui_app
heroku container:push web -a e2e-model-development
heroku container:release web -a e2e-model-development
```