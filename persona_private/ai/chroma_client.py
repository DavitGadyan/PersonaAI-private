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
from langchain.schema import Document

class StringLoader:
    def __init__(self, text: str, metadata: str):
        self.text = text
        self.metadata = metadata

    def load(self):
        # Create a Document from the string
        return [Document(page_content=self.text, metadata=self.metadata)]
    
load_dotenv()


def load_docs(filepath):
    '''Read documents from json object and split them into chunks

    Args:
        filepath (str): path to files on local machine
    '''
    # loader = DirectoryLoader(path, glob="./*.json", show_progress=True) #, loader_cls=TextLoader
    # documents = loader.load()

    # file_path='./file-2024.08.20.13.28.json'
    data = json.loads(Path(filepath).read_text())

    json_text = ""
    for country in data.keys():
        for feature in data[country].keys():
            json_text += f'\n For Country: {country} feature {feature} values are {str(data[country][feature])}'
        json_text += "\n\n"

    # loader = JSONLoader(
    #         file_path=filepath,
    #         jq_schema='.',
    #         text_content=False)

    loader = StringLoader(json_text, metadata={"source": filepath.split('/')[-1]})
    documents = loader.load()

    text_splitter = CharacterTextSplitter (chunk_size=1024, chunk_overlap=50)
    documents = text_splitter.split_documents(documents)

    return documents

def load_json(json_obj, filename):
    '''Read documents from json object and split them into chunks

    Args:
        json_obj (dict): json object
        filename (str): name of file
    '''

    data = json_obj
    out_l = []
    
    for country in data.keys():
        json_text = ""
        for feature in data[country].keys():
            json_text += f'\n For Country: {country} feature {feature} values are {str(data[country][feature])}'
        json_text += "\n\n"

        loader = StringLoader(json_text, metadata={"source": filename})
        documents = loader.load()

        text_splitter = CharacterTextSplitter (chunk_size=102400, chunk_overlap=50)
        documents = text_splitter.split_documents(documents)
        out_l.append(documents)

    return [item for sublist in out_l for item in sublist]



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
    print('n_documents>>', len(vectorstore.get()['documents']))
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
