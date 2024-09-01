# Import libraries
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
import os
import json
#Ollama
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import JSONLoader
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()



def load_docs(filepath):
    '''Read documents from json object and split them into chunks

    Args:
        filepath (str): path to files on local machine
    '''
    # loader = DirectoryLoader(path, glob="./*.json", show_progress=True) #, loader_cls=TextLoader
    # documents = loader.load()

    # file_path='./file-2024.08.20.13.28.json'
    # data = json.loads(Path(file_path).read_text())
    loader = JSONLoader(
            file_path=file_path,
            jq_schema='.',
            text_content=False)

    documents = loader.load()

    text_splitter = CharacterTextSplitter (chunk_size=1024, chunk_overlap=50)
    documents = text_splitter.split_documents(documents)

    return documents

def doc2chroma(docs, persist_directory):
    '''Save embeddings to Chroma

    Args:
        docs (Docs): company documents
        persist_directory (str): directory to store files as vectors

    '''

    ## set mistral embeddings
    # Ollama embeddings
    embeddings_model = OllamaEmbeddings(model="mistral", base_url='http://0.0.0.0:11434',)
    


    ## set chroma vectorstore
    vectorstore = Chroma.from_documents(documents=docs,
                                 # Chose the embedding you want to use
                                 # embedding=embeddings_open,
                                 embedding=embeddings_model,   
                                 persist_directory=persist_directory)

    vectorstore.persist()
    retriever = vectorstore.as_retriever()

    return retriever

def  save_docs2_chroma(path):
    docs = load_docs(path)
    retriever = doc2chroma(docs, persist_directory="docs_chromadb")
    return retriever

def get_retriever(persist_directory="docs_chromadb"):
    '''Get retriever object from Chroma

    Args:
        persist_directory (str): name of database
    '''
    embeddings_model = OllamaEmbeddings(model="mistral", base_url='http://0.0.0.0:11434',)
    vectorstore = Chroma(persist_directory=persist_directory,
                        embedding_function=embeddings_model
                        )
    
    retriever = vectorstore.as_retriever()

    return retriever
