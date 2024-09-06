import re
import os
import shutil
import datetime

from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders import NotebookLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from utils import load_docs_from_jsonl, save_docs_to_jsonl

load_dotenv() 


def load_documents_process_vectorize(todo_documents_path, past_documents_path):
    total_docs = []

    for filename in os.listdir(todo_documents_path):
        file_path = os.path.join(todo_documents_path, filename)

        if filename.endswith('.txt'):
            loader = TextLoader(file_path)
            docs = loader.load()
            total_docs.extend(docs)

        elif filename.endswith('.pdf'):
            loader = PDFMinerLoader(file_path)
            docs = loader.load()
            docs[0].page_content = docs[0].page_content.replace('\x0c', " ")
            docs[0].page_content = docs[0].page_content.replace('\n', " ")
            docs[0].page_content = re.sub(r'\s{2,}', " ", docs[0].page_content)
            docs[0].page_content = docs[0].page_content.strip()
            total_docs.extend(docs)

        elif filename.endswith('.ipynb'):
            loader = NotebookLoader(file_path, include_outputs=False, remove_newline=True)
            docs = loader.load()
            total_docs.extend(docs)

        shutil.move(file_path, os.path.join(past_documents_path, filename))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=256)
    splits = text_splitter.split_documents(total_docs)
    for doc in splits:
        doc.page_content = "Source : " + os.path.basename(doc.metadata["source"]) + "\n" + doc.page_content

    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(model = "text-embedding-3-large"))
    
    return splits, vectorstore


def save(faiss_vectorstore_path, doc_path, past_database):
    # "index.faiss"와 "index.pkl" 파일이 존재하는지 확인
    index_faiss = os.path.join(faiss_vectorstore_path, "index.faiss")
    index_pkl = os.path.join(faiss_vectorstore_path, "index.pkl")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    
    if os.path.exists(index_faiss) and os.path.exists(index_pkl):
        # 기존 벡터 스토어 로드
        past_vector_store = FAISS.load_local(
            faiss_vectorstore_path, embeddings=OpenAIEmbeddings(model="text-embedding-3-large"), allow_dangerous_deserialization=True
        )
        vectorstore.merge_from(past_vector_store)
        
        # 현재 날짜-시간으로 폴더 생성
        backup_dir = os.path.join(past_database, timestamp)
        os.makedirs(backup_dir, exist_ok=True)
        
        # 기존 인덱스 파일 이동
        shutil.move(index_faiss, os.path.join(backup_dir, "index.faiss"))
        shutil.move(index_pkl, os.path.join(backup_dir, "index.pkl"))
    
    # 벡터 스토어 저장
    vectorstore.save_local(faiss_vectorstore_path)

    # doc_path에 doc.jsonl이 존재하는지 확인
    doc_jsonl = os.path.join(doc_path, "doc.jsonl")
    if os.path.exists(doc_jsonl):
        past_docs = load_docs_from_jsonl(doc_jsonl)
        docs.extend(past_docs)
        
        # 백업 디렉토리 생성 및 기존 jsonl 파일 이동
        backup_dir = os.path.join(past_database, timestamp)
        os.makedirs(backup_dir, exist_ok=True)
        shutil.move(doc_jsonl, os.path.join(backup_dir, "doc.jsonl"))

    save_docs_to_jsonl(docs, doc_jsonl)


docs, vectorstore = load_documents_process_vectorize("todo_documents", "past_documents")
save("faiss_db","docs","backup")