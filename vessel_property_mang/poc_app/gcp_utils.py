# Google cloud storage import
import streamlit as st
import os 
from dotenv import load_dotenv 
load_dotenv()

def get_create_bucket(storage_client,bucket_name=os.getenv("DOCS_GCP_BUCKET_NAME")):
    # Check if the bucket exists
    bucket = storage_client.lookup_bucket(bucket_name)

    if bucket is None:
        # Create a new bucket
        bucket = storage_client.create_bucket(bucket_name)
        st.write(f'Bucket {bucket_name} created.')
    else:
        st.write(f'Bucket {bucket_name} already exists.')

    return bucket 

def upload_to_gcp(uploaded_file,bucket,metadata=None):
    """Upload file to Google Cloud Storage."""
    blob = bucket.blob(uploaded_file.name)
    if metadata:
        blob.metadata = metadata 
    blob.upload_from_file(uploaded_file)
    return blob.public_url  


