import streamlit as st
import requests
import pandas as pd
from io import StringIO

# Set Streamlit page title
st.title("Model Prediction")

# File upload section
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])



if uploaded_file is not None:
    # Display the uploaded file
    st.write("Uploaded File:")
    df = pd.read_csv(uploaded_file, index_col=0)
    st.write(df)

    # Send the file to FastAPI for prediction
    try:
        # Rewind the uploaded file and prepare it for the POST request
        uploaded_file.seek(0)  # Reset the file pointer to the beginning
        response = requests.post(
            "http://localhost:8000/predict/",  # FastAPI endpoint
            files={"file": ("uploaded_file.csv", uploaded_file, "text/csv")}  # Set the correct content-type
        )

        # If the response is successful, display the predictions
        if response.status_code == 200:
            predictions = response.json()["predictions"]
            predictions_df = pd.DataFrame(predictions.values(), index = predictions.keys())
            st.write("Predictions:", predictions_df)
        else:
            st.error(f"Error: {response.json()['error']}")
    except Exception as e:
        st.error(f"Failed to connect to the API: {str(e)}")
