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
        print(f'Bucket {bucket_name} created.')
    else:
        print(f'Bucket {bucket_name} already exists.')

    return bucket 

def upload_to_gcp(uploaded_file, gcp_bucket, metadata=None):
    # Reset the stream position to the beginning
    uploaded_file.seek(0)
    blob = gcp_bucket.blob(uploaded_file.name)
    if metadata:
        blob.metadata = metadata 
    blob.upload_from_file(uploaded_file)
    return blob.public_url  


def get_uploaded_file_names(storage_client,bucket_name=os.getenv("DOCS_GCP_BUCKET_NAME")):
    blobs = storage_client.list_blobs(bucket_name)
    file_names = [blob.name for blob in blobs]
    return file_names
