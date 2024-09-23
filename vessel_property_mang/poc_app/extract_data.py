from models import TasksInstructions 
import os 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import spacy
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4
from langchain_core.documents import Document


openai_api_key = os.getenv("OPENAI_API_KEY")

def extract_store_instructions(text):

    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
    structured_llm = llm.with_structured_output(TasksInstructions)
    output = structured_llm.invoke(text)

    if output:
        formatted_list = []
        if output.tasks_and_instructions:
            # Process each TaskInstruction
            for task_instruction in output.tasks_and_instructions:
                # Join the task and instructions lists into strings
                task_str = ' '.join(task_instruction.task)
                instructions_str = ' '.join(task_instruction.instructions)
                
                # Format the string as desired
                formatted_string = f"task: {task_str} + instructions: {instructions_str}"
                
                # Add to the list
                formatted_list.append(formatted_string)
            
            pc = Pinecone(api_key= os.getenv("PINECONE_API_KEY"))   
            index = pc.Index('instructions-index')

            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            try:
                vector_store = PineconeVectorStore(index=index, embedding=embeddings)
                documents = [Document(page_content=text) for text in formatted_list]
                uuids = [str(uuid4()) for _ in range(len(documents))]

                vector_store.add_documents(documents=documents, ids=uuids)
                return True
            except Exception as e:
                print(f"error: {e}")
                return None
    return False
   


def extract_building_names(text,roberta_nlp):
    doc = roberta_nlp(text)
    buildings = [ent.text for ent in doc.ents if ent.label_ in ['FAC',"building","property"]]
    return list(set(buildings))