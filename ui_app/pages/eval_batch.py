import requests
import json
import random

import streamlit as st
import base64

import pandas as pd
from sklearn.metrics import classification_report


# Title
st.title('MNIST Digit BATCH Classification')

# File uploader
uploaded_file = st.file_uploader("Choose a PNG image...", type="png", 
                                 accept_multiple_files=True)

if isinstance(uploaded_file, list):
    st.write(f'Number of images uploaded: {len(uploaded_file)}. Showing samples:')
    name_list = [uf.name for uf in uploaded_file]
    image_list_base64 = [base64.b64encode(uf.getvalue()) for uf in uploaded_file]
    
    random.seed(42)
    if len(uploaded_file) > 20:
        uploaded_file_sample = random.sample(uploaded_file, 20)
    else:
        uploaded_file_sample = uploaded_file
    name_list_sample = [uf.name for uf in uploaded_file_sample]
    st.image(uploaded_file_sample, caption=name_list_sample, use_column_width=False)
    
    url = "http://127.0.0.1:8000/predict_batch/"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }

    data = {
        "images": [
            {
                "image": image_base64.decode('utf-8'),
                "name": name
            } for image_base64, name in zip(image_list_base64, name_list)
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    # # Show the data sent to the API, make it collapsible
    # with st.expander("Show data sent to API"):
    #     st.code(data)
    prediction = response.json()['prediction']

    # Display the prediction
    st.write(f'Prediction:')
    st.code(prediction)
    
    # Display the response time
    st.code(f'Response time: {response.elapsed.total_seconds()} seconds')

    # Create a button to download the prediction
    st.download_button(
        label="Download Prediction",
        data=json.dumps(prediction),
        file_name="prediction.json",
        mime="application/json"
    )
    
    if st.button("Analyze prediction"):
        # Display the prediction
        df_results = pd.DataFrame(prediction, columns=['name', 'y_pred'])
        df_results['y_true'] = [int(name.split('_')[0]) for name in df_results['name']]
        st.code(classification_report(df_results['y_true'], df_results['y_pred']))