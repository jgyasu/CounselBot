from langchain_community.document_loaders import PyMuPDFLoader
import logging
from langchain_community.vectorstores import Chroma
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_and_store_documents(path):
    documents = []
    for file in os.listdir(path):
        if file.endswith('.pdf'):
            try:
                loader = PyMuPDFLoader(os.path.join(path, file))
                documents.extend(loader.load())
            except Exception as e:
                logging.error(f"Error loading {file}: {e}")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    modelPath = "sentence-transformers/all-mpnet-base-v2"

    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath, 
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': False}
    )
    
    vector_db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
    vector_db.persist()
    return vector_db

vector_db = load_and_store_documents("./documents")