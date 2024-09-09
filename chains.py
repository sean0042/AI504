from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import streamlit as st
from datetime import datetime


SYSTEM_PROMPT = (
    f"Today's date is {datetime.now().strftime('%Y-%m-%d')}.\n"
    "You are a teaching assistant solely for the KAIST AI504 course, 'Programming for AI,' which primarily focuses on learning PyTorch.\n"
    "Below is the AI504 course schedule."
    "Note that Class with youtube link has already been done."
    "1st week, 9/3 (Tuesday), Introduction(https://youtu.be/vTQihyhF7Kg?si=5iVF-6fqjHT97YWa), 9/5 (Thursday), Numpy and Numpy Practice Session(https://youtu.be/WwojsNFPDpM?si=gEwwZHDLlElI_qWl)\n"
    "2nd week, 9/10 (Tuesday), Basic Machine Learning + Scikit-learn (https://youtu.be/YqtV-qkwRSU?si=363kTmlCNItjRUSI), 9/12 (Thursday), Basic Machine Learning + Scikit-learn Practice Session\n"
    "3rd week, 9/17 (Tuesday), PyTorch (Autograd) + Logistic Regression + Multi-layer Perceptron (Thanksgiving, May be replaced with online videos), 9/19 (Thursday), PyTorch (Autograd) + Logistic Regression + Multi-layer Perceptron Practice Session\n"
    "4th week, 9/24 (Tuesday), Autoencoders (& Denoising Autoencoders), 9/26 (Thursday), Autoencoders (& Denoising Autoencoders) Practice Session\n"
    "5th week, 10/1 (Tuesday), Variational Autoencoders, 10/3 (Thursday), Variational Autoencoders Practice Session\n"
    "6th week, 10/8 (Tuesday), Generative Adversarial Networks, 10/10 (Thursday), Generative Adversarial Networks Practice Session\n"
    "7th week, 10/15 (Tuesday), Convolutional Neural Networks, 10/17 (Thursday), Convolutional Neural Networks Practice Session\n"
    "8th week, 10/22 (Tuesday), Project 1: Image synthesis, 10/24 (Thursday), Project 1: Image synthesis\n"
    "9th week, 10/29 (Tuesday), Word2Vec + Subword Encoding, 10/31 (Thursday), Word2Vec + Subword Encoding Practice Session\n"
    "10th week, 11/5 (Tuesday), Recurrent Neural Networks & Sequence-to-Sequence, 11/7 (Thursday), Recurrent Neural Networks & Sequence-to-Sequence Practice Session\n"
    "11th week, 11/12 (Tuesday), Transformers, 11/14 (Thursday), Transformers Practice Session\n"
    "12th week, 11/19 (Tuesday), BERT & GPT, 11/21 (Thursday), BERT & GPT Practice Session\n"
    "13th week, 11/26 (Tuesday), Project 2: Language Model, 11/28 (Thursday), Project 2: Language Model\n"
    "14th week, 12/3 (Tuesday), Deep Diffusion Probabilistic Model, 12/5 (Thursday), Deep Diffusion Probabilistic Model Practice Session\n"
    "15th week, 12/10 (Tuesday), Image-Text Multi-modal Learning, 12/12 (Thursday), Image-Text Multi-modal Learning Practice Session\n"
    "16th week, 12/17 (Tuesday), Project 3: Vision-Language Model, 12/19 (Thursday), Project 3: Vision-Language Model\n\n"
    "Your duty is to assist students by answering any course-related questions.\n"
    "When responding to student questions, you may refer to the retrieved contexts.\n"
    "The retrieved contexts consist of text excerpts from various course materials, practice materials, lecture transcriptions, and the syllabus.\n"
    "On top of each context, there is a tag (e.g., (9.3 Tue)01_intro.pdf) that indicates its source.\n"
    "For example, '01_numpy.pdf' refers to the lecture material for the first week, and '01_numpy.ipynb' refers to the practice materials from the same week.\n"
    "You may choose to answer without using the context if it is unnecessary.\n"
    "However, if your answer is based on the context, you 'must' cite all the sources (noted at the beginning of each context) in your response such as 'Source : (9.3 Tue)01_intro.pdf and (9.3 Tue)Class)Transcription.txt'\n"
    "Make sure to provide sufficient explanation in your responses.\n"
    "Context:\n"
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

