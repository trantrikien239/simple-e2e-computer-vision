# Features
# - Upload a PNG image
# - Call a POST API to get the classification of the image
# - Display the uploaded image and the prediction

import requests
import json

import streamlit as st
import base64

# Title
st.title('MNIST Digit Classification')

# File uploader
uploaded_file = st.file_uploader("Choose a PNG image...", type="png")

# Display the uploaded image
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=False)
    image_base64 = uploaded_file.getvalue()
    image_base64 = base64.b64encode(image_base64).decode('utf-8')
    
    url = "http://127.0.0.1:8000/predict/"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    data = {
        "image": image_base64,
        "name": uploaded_file.name
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    # Show the data sent to the API, make it collapsible
    with st.expander("Show data sent to Prediction API"):
        st.code(data)
    prediction = response.json()['prediction']

    # Display the prediction
    st.write(f'Prediction: {prediction}')
    
    # Display the response time
    st.code(f'Response time: {response.elapsed.total_seconds()} seconds')
