import os 
from dotenv import load_dotenv 
load_dotenv()
from langchain_community.document_loaders.mongodb import MongodbLoader
from langchain_pinecone import PineconeVectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import S3FileLoader
from langchain_core.documents import Document
import logging
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import streamlit as st

def get_pinecone_vector_store(embeddings_model,namespace="default",index_name: str = st.secrets("PINECONE_INDEX_NAME")) -> PineconeVectorStore:
    try:
        # Initialize Pinecone client
        pinecone_api_key = st.secrets("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("Pinecone API key is not set. Please set it in the environment variables.")
        
        pc = Pinecone(api_key=pinecone_api_key)

        # Check if the index exists
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        if index_name in existing_indexes:
            print(f"Index '{index_name}' already exists. Retrieving it...")
            index = pc.Index(index_name)
        else:
            print(f"Index '{index_name}' does not exist. Please create one")


        # Initialize and return the vector store
        vector_store = PineconeVectorStore(index=index, embedding=embeddings_model,namespace=namespace)
        return vector_store

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

def chunk_docs(documents, embed_model) :
    """
    Chunks the provided documents into smaller text segments using a semantic chunking model.

    Args:
        documents (List[Document]): A list of documents to be chunked. Each document is expected to have a `page_content` attribute.
        embed_model: The embedding model to be used by the semantic chunker.

    Returns:
        List[Document]: A list of chunks created from the input documents.

    Raises:
        Exception: If chunking fails.
    """
    
    try:
        logging.info("Initializing SemanticChunker...")
        text_splitter = SemanticChunker(embeddings=embed_model)
        logging.info("Performing document chunking...")
        chunks = text_splitter.transform_documents(documents=documents)
        logging.info(f"Created {len(chunks)} chunks from the documents.")
        return chunks
    except Exception as e:
        logging.error(f"Failed to chunk documents: {str(e)}")
        raise

def load_to_vectordb(
    docs,
    index_name: str,
    embeddings,
    namespace: str = "default" 
) :
    """
    Loads the given documents into a vector database, using the specified embeddings and namespace.

    Args:
        docs (List[Document]): A list of document chunks to be loaded into the vector database.
        index_name (str): The name of the index in the vector database.
        dimension (int): The dimensionality of the vectors. Defaults to 1024.
        metric (str): The similarity metric to use. Defaults to "cosine".
        namespace (str): The namespace under which the documents are to be stored in the vector database.

    Returns:
        Tuple[Union[PineconeVectorStore, None], bool]: 
            - If successful: (vector_store_object, True)
            - If failed: (None, False)

    Raises:
        Exception: If loading to the vector database fails.
    """
    
    try:
        # Retrieve or create the Pinecone vector store
        vector_store = get_pinecone_vector_store(embeddings,namespace=namespace, index_name=index_name)
        
        if vector_store is None:
            logging.error("Failed to create or retrieve the Pinecone vector store.")
            return None, False

        logging.info(f"Loading {len(docs)} documents into Pinecone vector store '{index_name}' under namespace '{namespace}'...")
        
        # TODO: add cache 
        vector_store.add_documents(documents=docs)

        logging.info("Documents successfully loaded into the vector database.")
        return vector_store, True

    except Exception as e:
        logging.error(f"Failed to load documents into vector database: {str(e)}")
        return None, False
    

def load_document():

    loader = MongodbLoader(
        connection_string=st.secrets("MONGODB_CONNECTION_STRING"),
        db_name="streamlit-documents",
        collection_name="user-documents",
    )

    docs = loader.load()
    return docs


def process_documents(embeddings_model, index_name, namespace="default"):
    """
    Orchestrates the process of loading documents from MongoDB, chunking them, and then loading them into a Pinecone vector database.

    Args:
        embeddings_model: The embedding model used for chunking and creating vector representations of the documents.
        index_name (str): The name of the index in Pinecone where the vectors will be stored.
        namespace (str): The namespace in Pinecone under which the vectors will be stored. Defaults to "default".
    """
    try:
        # Step 1: Load documents from MongoDB
        logging.info("Loading documents from MongoDB...")
        documents = load_document()

        if not documents:
            logging.error("No documents found in MongoDB collection.")
            return

        # Step 2: Chunk the documents
        logging.info("Chunking the documents...")
        document_chunks = chunk_docs(documents, embeddings_model)

        if not document_chunks:
            logging.error("Failed to chunk documents.")
            return

        # Step 3: Load document chunks into Pinecone vector database
        logging.info("Loading document chunks into the vector database...")
        vector_store, success = load_to_vectordb(
            docs=document_chunks,
            index_name=index_name,
            embeddings=embeddings_model,
            namespace=namespace
        )

        if success:
            logging.info("Documents successfully processed and loaded into Pinecone.")
            return vector_store
        else:
            logging.error("Failed to load document chunks into the vector database.")
            return None
    
    except Exception as e:
        logging.error(f"An error occurred during document processing: {str(e)}")
