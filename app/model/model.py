import pickle
import re
from pathlib import Path
import pandas as pd 
import os
import io
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"external_resources"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"external_resources/prep"))
from prep.preprocessing import FeatureExtractionTransformer, DropUnselectedColumns, EncodeOrganizationType, ReorderColumns, DecodeCatOrdEncoding
from google.cloud import storage
import os

# Path to your service account key
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\M3_capstone\gkm3capstone-banking-project-7ac9a260a17b.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """
    Download a file from GCS bucket.
    
    :param bucket_name: Name of the GCS bucket.
    :param source_blob_name: Name of the file in the bucket.
    :param destination_file_name: Local path to save the file.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    # Download the file
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}.")


__version__ = "0.1"

BASE_DIR = Path(__file__).resolve(strict=True).parent

download_from_gcs("gk_m3_capstone_bucket", "model.pkl", "model.pkl")
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


def load_and_prep_data(per_data):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "external_resources")

    with open(os.path.join(DATA_DIR, 'engineered_and_selected.pkl'), 'rb') as f:
        engineered_and_selected = pickle.load(f)

    with open(os.path.join(DATA_DIR,"ordinal_enc_cols.pkl"), "rb") as f:
        ordinal_enc_cols = pickle.load(f)
        
    with open(os.path.join(DATA_DIR,"one_hot_enc_cols.pkl"), "rb") as f:
        one_hot_enc_cols = pickle.load(f)
        
    with open(os.path.join(DATA_DIR,"categories.pkl"), "rb") as f:
        categories = pickle.load(f)
        
    with open(os.path.join(DATA_DIR,"rank_selected_columns.pkl"), "rb") as f:
        rank_selected_columns = pickle.load(f)

    with open(os.path.join(DATA_DIR,"VIF_drop.pkl"), "rb") as f:
        VIF_drop = pickle.load(f)

    with open(os.path.join(DATA_DIR,"cols_to_scale.pkl"), "rb") as f:
        cols_to_scale = pickle.load(f)


    #installments_payments = pd.read_csv(os.path.join(DATA_DIR, "installments_payments.csv"), index_col=0)
    #bureau = pd.read_csv(os.path.join(DATA_DIR, "bureau.csv"), index_col=1)
    #application_prev = pd.read_csv(os.path.join(DATA_DIR, "previous_application.csv"), index_col=0)
    #POS_CASH_balance = pd.read_csv(os.path.join(DATA_DIR, "POS_CASH_balance.csv"), index_col=0)
    download_from_gcs("gk_m3_capstone_bucket", "fitted_preprocessig_pipeline.pkl", "fitted_preprocessig_pipeline.pkl")
    with open('fitted_preprocessig_pipeline.pkl', 'rb') as f:
        prerocess_pipeline = pickle.load(f)
    
    #file_like_object = io.BytesIO(per_data)
    #per_data_df =  pd.read_csv(file_like_object, index_col=0)
    transformed_data = prerocess_pipeline.transform(per_data)

    with open(os.path.join(DATA_DIR,"final_columns.pkl"), "rb") as f:
        final_columns = pickle.load(f)
    #with open(os.path.join(DATA_DIR,'final_columns.txt'), 'rb') as f:
    #    final_columns = [line.strip() for line in f]

    return transformed_data[final_columns]

def predict_pipeline(df):
    df_prepped = load_and_prep_data(df)
    pred = model.predict(df_prepped)
    return pred
