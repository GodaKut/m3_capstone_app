import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fastapi import FastAPI
from pydantic import BaseModel
from model.model import predict_pipeline
from model.model import __version__ as model_version
import pandas as pd 
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pickle
from io import StringIO


app = FastAPI()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
origins = [
    "http://localhost:8501",  # Default Streamlit port
    "http://127.0.0.1:8501",  # Localhost address
    # You can add more allowed origins if needed (e.g., production URLs)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}

@app.post("/predict/")
#@app.post("/predict", response_model=PredictionOut)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # If it's a CSV file, we can process it
    if file.content_type == 'text/csv':
        try:
            # Convert the bytes to a StringIO object and read it into a DataFrame
            data = StringIO(contents.decode("utf-8"))
            df = pd.read_csv(data, index_col=0)
            
            # Make predictions using the model
            # Here, assuming the model expects the DataFrame to be in a specific format
            predictions = predict_pipeline(df)
            result_class = {1: "Risk of Payment difficulties", 0: "No risk"}
            predictions_dir = {}
            for i, df_index in enumerate(df.index):
                predictions_dir[str(df_index)] = result_class[int(predictions[i])]

            # Return the predictions
            return JSONResponse(content={"predictions": predictions_dir})

        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Error processing file: {str(e)}"})
    else:
        return JSONResponse(status_code=400, content={"error": "File format not supported. Please upload a CSV file."})
