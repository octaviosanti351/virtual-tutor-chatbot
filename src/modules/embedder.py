import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from typing import List
from langchain.docstore.document import Document
import os
import glob



DEFAULT_VECTOR_STORE_PATH = '/home/bitlogic/IdeaProjects/virtual-tutor-chatbot/embeddings'

def ingest_docs():

    document = '/home/bitlogic/IdeaProjects/virtual-tutor-chatbot/documents'
    embeddingsPath = '/home/bitlogic/IdeaProjects/virtual-tutor-chatbot/embeddings'
    file_extension = ".faiss"

    file_pattern = os.path.join(embeddingsPath, f"*{file_extension}")
    matching_files = glob.glob(file_pattern)

    if not matching_files:
        loader = DirectoryLoader(document, glob="**/*.txt")
        raw_documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
        )

        documents = text_splitter.split_documents(raw_documents)

        print(f"Going to add {len(documents)} to faiss db")
        save_embeddings(documents, "canvas-bot")
    else:
        print("Files already downloaded")


# save_embeddings should be an interface to work as a port for different vector store
def save_embeddings(documents: List[Document], index: str, vector_store_path: str = DEFAULT_VECTOR_STORE_PATH):
    embeddings = OpenAIEmbeddings(openai_api_key="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")

    vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
    vector_store.save_local(folder_path=vector_store_path, index_name=index)

