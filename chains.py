
from langchain_core.messages import ChatMessage,HumanMessage,AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.tracers.context import collect_runs
from streamlit_feedback import streamlit_feedback
import streamlit as st
from langsmith import Client
from langsmith import traceable


system_prompt = (
    "You are a teaching assistant for the KAIST AI504 course 'Programming for AI,' which focuses on learning NumPy and PyTorch."
    "You may use the following retrieved context to answer the student's question if relevant."
    "The retrieved context consists of course materials, practice materials, class audio transcriptions, and the syllabus."
    "For example, '01_numpy.pdf' refers to the first week's lecture materials, and '01_numpy.ipynb' refers to the first week's practice materials."
    "However, if the context is not necessary to answer the question, you may choose not to use it."
    "If you do use the context, you must cite the source (noted at the beginning of each context) in your response.\nContext :"
)


def get_vector_store():
    vector_store= FAISS.load_local(
        "./faiss_db/20240902", 
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"), 
        allow_dangerous_deserialization=True)
    return vector_store



def get_retreiver_chain(vector_store):
  llm=ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
  retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3},
  )
  prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user","{input}"),
      ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
  ])
  history_retriver_chain = create_history_aware_retriever(llm,retriever,prompt)
  return history_retriver_chain



def get_conversational_rag(history_retriever_chain):
  llm=ChatOpenAI(model = "gpt-4o", temperature = 0)
  answer_prompt=ChatPromptTemplate.from_messages([
      ("system",system_prompt+"\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user","{input}")
  ])

  document_chain = create_stuff_documents_chain(llm,answer_prompt)
  conversational_retrieval_chain = create_retrieval_chain(history_retriever_chain,document_chain)
  return conversational_retrieval_chain

