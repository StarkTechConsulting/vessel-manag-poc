from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os 
from dotenv import load_dotenv
load_dotenv()
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run_query( retriever,query):

    llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=st.secrets["OPENAI_API_KEY"]
)
    
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. provide as many details as powwible in your "
        "answer concise.Important: return just the final answer and do not include words like 'response:' or similar expressions"
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
   

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})
    return response["answer"]