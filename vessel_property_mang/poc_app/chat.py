from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os 
from dotenv import load_dotenv
load_dotenv()
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from etl import get_pinecone_vector_store
from langchain_community.agent_toolkits import GmailToolkit
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import OpenAIEmbeddings
history = StreamlitChatMessageHistory(key="chat_messages")




def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run_query( retriever,query,chat_history):

    llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=st.secrets["OPENAI_API_KEY"]
    )
    


    # Contextualize question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Answer question
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Use three sentences maximum and keep the answer "
        "concise."
        "{context}"
    ) 
    qa_prompt = ChatPromptTemplate.from_messages([("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}"),])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt) 
    rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain
    )
    
    answer = rag_chain.invoke({"input": query, "chat_history": chat_history})
    history.add_user_message(query)
    history.add_ai_message(answer)

    return answer


def run_agent(query,chat_history,memory):

    openai_api_key=os.getenv("OPENAI_API_KEY")
    toolkit = GmailToolkit()

    # Get tools
    google_tools = toolkit.get_tools()
    vector_store = get_pinecone_vector_store(OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"),model="text-embedding-3-large"))
    docs_retriever = vector_store.as_retriever()

    inst_vector_store = get_pinecone_vector_store(OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"),model="text-embedding-3-large"),index_name="instructions-index")
    inst_retriever = vector_store.as_retriever()

    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)

    docs_retrieval_tool = create_retriever_tool(
        docs_retriever,
        "answer_question",
        "Invoked always when a user asks a question."
    )

    instructions_retrieval_tool = create_retriever_tool(
        inst_retriever,
        "instructions_retriever",
        "Used to retrieve relevant instructions on how to complete a task when the user asks you to perform a certain task like letter writting, email drafting etc."
    )
    tools = [docs_retrieval_tool,instructions_retrieval_tool] + google_tools



    # OpenAI functions agent
    instructions = "You are an assistant."
    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    prompt = base_prompt.partial(instructions=instructions)
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
    openai_agent = create_openai_functions_agent(llm, tools, prompt)

    # MEMORY VS CHAT HISTORY?
    agent_executor = AgentExecutor(
        agent=openai_agent,
        tools=tools,
        verbose=True,
        memory=memory
    )
    output = agent_executor.invoke(  {
        "input": query,
        "chat_history": chat_history
    }
    )

    return output["output"]
