import pickle
from pathlib import Path
import os
from google.cloud import storage
import sys
#current_dir = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(current_dir)
import preprocessing

# Path to your service account key

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

download_from_gcs("gk_m3_capstone_bucket", "model.pkl", (os.path.join(BASE_DIR,"model.pkl")))
with open(os.path.join(BASE_DIR,"model.pkl"), "rb") as f:
    model = pickle.load(f)


def load_and_prep_data(per_data):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    download_from_gcs("gk_m3_capstone_bucket", "fitted_preprocessig_pipeline.pkl", (os.path.join(BASE_DIR,"fitted_preprocessig_pipeline.pkl")))
    with open(os.path.join(BASE_DIR,'fitted_preprocessig_pipeline.pkl'), 'rb') as f:
        prerocess_pipeline = pickle.load(f)

    transformed_data = prerocess_pipeline.transform(per_data)

    download_from_gcs("gk_m3_capstone_bucket", "final_columns.pkl", (os.path.join(BASE_DIR,"final_columns.pkl")))
    with open(os.path.join(BASE_DIR,"final_columns.pkl"), "rb") as f:
        final_columns = pickle.load(f)


    return transformed_data[final_columns]

def predict_pipeline(df):
    df_prepped = load_and_prep_data(df)
    pred = model.predict(df_prepped)
    return pred
