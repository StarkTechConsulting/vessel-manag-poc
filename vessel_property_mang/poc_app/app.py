import streamlit as st
from pymongo import MongoClient
import os 
from dotenv import load_dotenv 
load_dotenv()
import gridfs
from chat import run_query
from bson.objectid import ObjectId
from langchain_openai import OpenAIEmbeddings
from etl import process_documents
from langchain_pinecone import PineconeVectorStore

client = MongoClient(st.secrets("MONGODB_CONNECTION_STRING"))  
db = client["streamlit-documents"]  
fs = gridfs.GridFS(db)  # GridFS is used to store large files in MongoDB
vector_store = PineconeVectorStore.from_existing_index(index_name=st.secrets("PINECONE_INDEX_NAME"), embedding=OpenAIEmbeddings(api_key=st.secrets("OPENAI_API_KEY"),model="text-embedding-3-large"))


def upload_to_mongo(file):
    file_id = fs.put(file, filename=file.name)
    return file_id

def get_file(file_id):
    grid_out = fs.get(ObjectId(file_id))
    return grid_out.read(), grid_out.filename


st.title("Document Upload & Query App")

# Section for file upload
st.header("Upload your documents")
uploaded_files = st.file_uploader("Choose documents", type=['pdf', 'txt', 'docx'], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_id = upload_to_mongo(uploaded_file)
        st.success(f"File {uploaded_file.name} uploaded successfully with ID {file_id}")

# Section for text query input
st.header("Ask a question about the uploaded documents")
user_query = st.text_area("Enter your query:")

if st.button("Submit Query"):
    if user_query:
        print(user_query)
        st.write("Processing your query...")
        response = run_query(vector_store,user_query)
        st.write("Response:", response)
    else:
        st.error("Please enter a query")

# Display uploaded documents with links
st.header("Uploaded Documents")
uploaded_docs = fs.find()  # Get all documents from GridFS

if uploaded_docs:
    st.write("Here are the uploaded documents:")
    for doc in uploaded_docs:
        file_id_str = str(doc._id)
        file_content, filename = get_file(file_id_str)

        # Use Streamlit's download_button to allow the user to download the file
        st.download_button(
            label=f"Download {filename}",
            data=file_content,
            file_name=filename,
            mime="application/octet-stream"
        )
else:
    st.write("No documents uploaded yet")

# Streamlit route to handle file download
@st.cache_data(show_spinner=False)
def serve_file(file_id):
    file_content, filename = get_file(file_id)
    return file_content, filename

if st.experimental_get_query_params().get("file_id"):
    file_id = st.query_params.get("file_id")[0]
    file_content, filename = serve_file(file_id)
    st.download_button(label=f"Download {filename}", data=file_content, file_name=filename)