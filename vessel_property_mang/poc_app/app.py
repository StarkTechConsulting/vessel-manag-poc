import streamlit as st
from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()
import gridfs
from chat import run_query
from bson.objectid import ObjectId
from langchain_openai import OpenAIEmbeddings
from etl import process_document, get_pinecone_vector_store
import tempfile

# Initialize MongoDB and GridFS
client = MongoClient(st.secrets["MONGODB_CONNECTION_STRING"])
db = client["streamlit-documents"]
fs = gridfs.GridFS(db)  # GridFS is used to store large files in MongoDB

# Initialize OpenAI embeddings model
embeddings_model = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"], model="text-embedding-3-large")

# Initialize session state for the vector store and processing status
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_pinecone_vector_store(embeddings_model=embeddings_model)

if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = True  # Set to True initially so querying is enabled before any uploads

def upload_to_mongo(file):
    """Upload file to MongoDB's GridFS."""
    file_id = fs.put(file, filename=file.name)
    return file_id

@st.cache_data(show_spinner=False)
def get_file(file_id):
    """Retrieve file from MongoDB by file ID."""
    grid_out = fs.get(ObjectId(file_id))
    return grid_out.read(), grid_out.filename

def save_temp_file(uploaded_file):
    """Save the uploaded file to a temporary file and return the file path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
        temp_file.write(uploaded_file.read())
        return temp_file.name  # Return the temporary file path

# Page 1: Question Answering Page Logic
def question_answering_page():
    st.title("Document Upload & Query App")
    
    # Section for file upload
    st.header("Upload your documents")
    uploaded_files = st.file_uploader("Choose documents", type=['pdf', 'txt', 'docx'], accept_multiple_files=True)

    if uploaded_files:
        # Disable querying while processing the uploaded documents
        st.session_state.processing_complete = False
        for uploaded_file in uploaded_files:
            # Save the uploaded file to MongoDB
            file_id = upload_to_mongo(uploaded_file)
            st.success(f"File {uploaded_file.name} uploaded successfully with ID {file_id}")
            
            # Save the file to a temporary location and process it
            temp_file_path = save_temp_file(uploaded_file)
            st.write(f"Processing document {uploaded_file.name}...")
            
            # Process the document and store the result in the vector store in session state
            with st.spinner("Processing..."):
                st.session_state.vector_store = process_document(temp_file_path,embeddings_model,st.secrets["PINECONE_INDEX_NAME"])
                st.session_state.processing_complete = True  # Enable querying after processing is complete
            
            st.success(f"Document {uploaded_file.name} processed successfully.")

    # Section for text query input
    st.header("Ask a question ")
    
    # Query input is enabled by default, but disabled while processing documents
    user_query = st.text_area("Enter your query:", disabled=not st.session_state.processing_complete)

    if st.button("Submit Query", disabled=not st.session_state.processing_complete):
        if user_query:
            # Ensure vector store is not None before calling as_retriever
            if st.session_state.vector_store is not None:
                st.write("Processing your query...")
                retriever = st.session_state.vector_store.as_retriever()  # Use the session-stored vector store
                response = run_query(retriever, user_query)
                st.write("Response:", response)
            else:
                st.error("Error: Vector store is not available. Please upload and process documents first.")
        else:
            st.error("Please enter a query")

# Page 2: Uploaded Documents Page Logic
def uploaded_documents_page():
    st.title("Uploaded Documents")
    
    # Display uploaded documents with download links
    uploaded_docs = fs.find()  # Get all documents from GridFS

    if uploaded_docs:
        st.write("Here are the uploaded documents:")
        seen_files = set()  # Track file names to avoid duplicates
        for doc in uploaded_docs:
            file_id_str = str(doc._id)
            file_content, filename = get_file(file_id_str)

            # Avoid duplicate documents in the listing
            if filename not in seen_files:
                seen_files.add(filename)

                # Use Streamlit's download_button to allow the user to download the file
                st.download_button(
                    label=f"Download {filename}",
                    data=file_content,
                    file_name=filename,
                    mime="application/octet-stream",
                    key=f"download_{file_id_str}"  # Unique key for each download button
                )
    else:
        st.write("No documents uploaded yet.")

# Main function to control navigation and page display
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Question Answering", "Uploaded Documents"])
    
    if page == "Question Answering":
        question_answering_page()
    elif page == "Uploaded Documents":
        uploaded_documents_page()

# Run the main function
if __name__ == "__main__":
    main()
