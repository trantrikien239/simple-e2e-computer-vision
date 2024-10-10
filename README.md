### Installing the environment

```bash
pip install -r requirements.txt
```

### Train

```bash
python model/train.py
```

### Deploy with Docker
```
./copy_files.sh
docker-compose up --build
```

This will build and start the Prediction API and the UI as separate services that are connected via a virtual network. Access locally:
    - Prediction API docs: `http://localhost:8000/docs`
    - Streamlit UI: `http://localhost:8501/`

![Prediction API - Backend](_assets/fastapi_backend.png)

![Streamlit Frontend](_assets/streamlit_frontend.png)