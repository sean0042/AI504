from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import streamlit as st


SYSTEM_PROMPT = (
    "You are a teaching assistant for the KAIST AI504 course, 'Programming for AI,' which primarily focuses on learning PyTorch."
    "Your duty is to assist students by answering any course-related questions."
    "When responding to student questions, you may refer to the retrieved contexts."
    "The retrieved contexts consist of text excerpts from various course materials, practice materials, lecture transcriptions, and the syllabus."
    "On top of each context, there is a tag (e.g., '01_numpy.pdf') that indicates its source."
    "For example, '01_numpy.pdf' refers to the lecture material for the first week, and '01_numpy.ipynb' refers to the practice materials from the same week."
    "You may choose to answer without using the context if it is unnecessary."
    "However, if you do use the context, you should cite the sources (noted at the beginning of each context) that you used in your response.\nContext:"
)


def get_vector_store():
    # Load a local FAISS vector store
    vector_store = FAISS.load_local(
        "./faiss_db/", 
        embeddings = OpenAIEmbeddings(model = "text-embedding-3-large"), 
        allow_dangerous_deserialization = True)
    
    return vector_store



def get_retreiver_chain(vector_store):

    llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)

    faiss_retriever = vector_store.as_retriever(
       search_kwargs={"k": 5},
    )
    # bm25_retriever = BM25Retriever.from_documents(
    #    st.session_state.docs
    # )
    # bm25_retriever.k = 2

    # ensemble_retriever = EnsembleRetriever(
    #     retrievers = [bm25_retriever, faiss_retriever],
    # )

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name = "chat_history"),
        ("user","{input}"),
        ("user","Based on the conversation above, generate a search query that retrieves relevant information. Provide enough context in the query to ensure the correct document is retrieved. Only output the query.")
    ])
    history_retriver_chain = create_history_aware_retriever(llm, faiss_retriever, prompt)

    return history_retriver_chain




def get_conversational_rag(history_retriever_chain):
  # Create end-to-end RAG chain
  llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)

  answer_prompt = ChatPromptTemplate.from_messages([
      ("system",SYSTEM_PROMPT+"\n\n{context}"),
      MessagesPlaceholder(variable_name = "chat_history"),
      ("user","{input}")
  ])

  document_chain = create_stuff_documents_chain(llm,answer_prompt)

  conversational_retrieval_chain = create_retrieval_chain(history_retriever_chain, document_chain)

  return conversational_retrieval_chain

