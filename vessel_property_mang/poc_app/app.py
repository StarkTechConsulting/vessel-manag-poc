import streamlit as st
from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()
import gridfs
from bson.objectid import ObjectId
import tempfile
from datetime import datetime

# LangChain imports
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.embeddings import OpenAIEmbeddings
from google.cloud import storage
from etl import process_document, get_pinecone_vector_store
from gcp_utils import get_create_bucket, upload_to_gcp
import spacy
from extract_data import extract_building_names, extract_store_instructions
from chat import run_query
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(os.getenv('SERVICE_ACCOUNT_PATH'))
storage_client = storage.Client(credentials=credentials)

gcp_bucket = get_create_bucket(storage_client,os.getenv("DOCS_GCP_BUCKET_NAME"))

client = MongoClient(st.secrets["MONGODB_CONNECTION_STRING"])
db = client["streamlit-documents"]
conversations_collection = db['chat-history']

# Get an OpenAI API Key before continuing
if "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
else:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Enter an OpenAI API Key to continue")
    st.stop()

# Initialize OpenAI embeddings model
embeddings_model = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-large")

# Initialize session state for the vector store and processing status
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_pinecone_vector_store(embeddings_model=embeddings_model)

if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = True  # Set to True initially so querying is enabled before any uploads



def save_temp_file(uploaded_file):
    """Save the uploaded file to a temporary file and return the file path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
        temp_file.write(uploaded_file.read())
        return temp_file.name  # Return the temporary file path

def load_conversation():
    """Load conversation history from MongoDB."""
    messages = []
    cursor = conversations_collection.find().sort('timestamp', 1)  # Sort by timestamp ascending
    for doc in cursor:
        role = doc['role']
        content = doc['content']
        if role == 'human':
            messages.append(HumanMessage(content=content))
        elif role == 'ai':
            messages.append(AIMessage(content=content))
    return messages

def save_message(role, content):
    """Save a single message to MongoDB."""
    conversations_collection.insert_one({
        'role': role,
        'content': content,
        'timestamp': datetime.utcnow()
    })



# Page 1: Question Answering Page Logic
def question_answering_page():
    st.title("Document Upload & Query App")
    
    # Section for file upload
    st.header("Upload your documents")
    uploaded_files = st.file_uploader("Choose documents", type=['pdf', 'txt', 'docx'], accept_multiple_files=True)

    if uploaded_files:
        roberta_nlp = spacy.load("en_core_web_trf")
        # Disable querying while processing the uploaded documents
        st.session_state.processing_complete = False

        for uploaded_file in uploaded_files:

            # Save the file to a temporary location and process it
            temp_file_path = save_temp_file(uploaded_file)
            st.write(f"Processing document {uploaded_file.name}...")
            
            # Process the document and store the result in the vector store in session state
            with st.spinner("Processing..."):
                st.session_state.vector_store,files_docs = process_document(
                    temp_file_path,
                    embeddings_model,
                    st.secrets["PINECONE_INDEX_NAME"]
                )
                st.session_state.processing_complete = True  # Enable querying after processing is complete
            
                        
            st.success(f"Document {uploaded_file.name} processed to vector store successfully.")

            doc_metadata = {"property":extract_building_names(files_docs,roberta_nlp)}

            # Save the uploaded file to MongoDB
            file_id = upload_to_gcp(uploaded_file,gcp_bucket,metadata=doc_metadata)
            st.success(f"File {uploaded_file.name} uploaded successfully with ID {file_id}")
            
           

    # Section for chatbot TODO: MODIFY TO AGENT
    st.header("Chat with your documents")
    
    # load conversation history
    if "messages" not in st.session_state:
        # Load messages from the database
        messages = load_conversation()
        st.session_state.messages = messages

    # Check if vector store is available
    if st.session_state.vector_store is not None:
        retriever = st.session_state.vector_store.as_retriever()
    else:
        st.error("No vector store found. Please upload and process documents first.")
        st.stop()
    
    
  
    print(messages) 

    for msg in messages:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)

    # Get user input
    if st.session_state.processing_complete:

        if user_input := st.chat_input("Enter your question"):

            st.chat_message("user").write(user_input)
            # extract instructions
            instructions_output = extract_store_instructions(user_input)
            if instructions_output:
                st.success("Successfully saved user instructions")

            response = run_query(retriever,user_input,st.session_state.messages) # RUN USER QUERY TODO: replace with agent
            answer = response["answer"]
            st.chat_message("assistant").write(answer)
            
            # Save the messages to the database
            save_message('human', user_input)
            save_message('ai', answer)
    else:
        st.info("Please upload and process documents to start chatting.")


# Page 2: Uploaded Documents Page Logic
def uploaded_documents_page(metadata_field,bucket_name= os.getenv("DOCS_GCP_BUCKET_NAME")):
    st.title("Uploaded Documents")
    
    # List all blobs in the GCS bucket
    blobs = storage_client.list_blobs(bucket_name)
    blob_list = list(blobs)
    
    if blob_list:
        # Use a defaultdict to group blobs by metadata field value
        from collections import defaultdict
        grouped_blobs = defaultdict(list)
        
        # Group blobs based on the metadata field
        for blob in blob_list:
            # Retrieve metadata
            metadata = blob.metadata or {}
            # Get the value of the specified metadata field
            field_value = metadata.get(metadata_field, 'Unknown')
            # Add blob to the appropriate group
            grouped_blobs[field_value].append(blob)
        
        # Display blobs grouped by the metadata field value
        for field_value, blobs_in_group in grouped_blobs.items():
            # Write a headline for each metadata value
            st.header(f"{field_value} docs:")
            for blob in blobs_in_group:
                # Get the file name without prefixes
                file_name = blob.name.split('/')[-1]
                
                # Download the file content
                file_content = blob.download_as_bytes()
                
                # Use Streamlit's download_button to allow the user to download the file
                st.download_button(
                    label=f"Download {file_name}",
                    data=file_content,
                    file_name=file_name,
                    mime=blob.content_type or "application/octet-stream",
                    key=f"download_{blob.name}"
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
        metadata_field = "property" # TODO: use consistently 
        uploaded_documents_page(metadata_field)

# Run the main function
if __name__ == "__main__":
    main()
