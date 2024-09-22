# Import libraries
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.embeddings import OpenAIEmbeddings
import os
import json
import time
#Ollama
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import JSONLoader
from pathlib import Path
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from persona_private.ai.agent import mistral7b_llm

class StringLoader:
    def __init__(self, text: str, metadata: str):
        self.text = text
        self.metadata = metadata

    def load(self):
        # Create a Document from the string
        return [Document(page_content=self.text, metadata=self.metadata)]
    
load_dotenv()

metadata_field_info = [
    AttributeInfo(
        name="CountryName",
        description="The country of the record",
        type="string",
    ),
    # AttributeInfo(
    #     name="year",
    #     description="The year the movie was released",
    #     type="integer",
    # ),
    # AttributeInfo(
    #     name="director",
    #     description="The name of the movie director",
    #     type="string",
    # ),
    # AttributeInfo(
    #     name="rating", description="A 1-10 rating for the movie", type="float"
    # ),
]

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

    text_splitter = CharacterTextSplitter (chunk_size=102400, chunk_overlap=50)
    documents = text_splitter.split_documents(documents)

    return documents

def load_json(json_obj, filename):
    '''Read documents from json object and split them into chunks

    Args:
        json_obj (dict): json object
        filename (str): name of file
    '''

    data = json_obj
    
    print("data>>", data.keys())
    json_text = ""
    for country in data.keys():
        for feature in data[country].keys():
            json_text += f'\n For Country: {country} feature {feature} values are {str(data[country][feature])}'
        json_text += "<END-END>"
    print(len(json_text.split("<END-END>")))
    time.sleep(10)
    loader = StringLoader(json_text, metadata={"source": filename})
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=2052, chunk_overlap=0, separator="<END-END>")
    documents = text_splitter.split_documents(documents)

    return documents

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["Country"] = record.get("CountryName")
    return metadata

def load_json2(json_obj, filename):
    '''Read documents from json object and split them into chunks

    Args:
        json_obj (dict): json object
        filename (str): name of file
    '''

    data = json_obj
    file_path='./data.json'

    for k, v in data.items():
        data[k]["CountryName"] = k

    with open(file_path, 'w') as f:
        json.dump(data, f)
    
    loader = JSONLoader(
            file_path=file_path,
            jq_schema='.[]',
            text_content=False,
            metadata_func=metadata_func)

    documents = loader.load()

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
                                collection_metadata={"hnsw:space": "cosine"},
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
    embeddings_model = OllamaEmbeddings(model="mistral", base_url='http://0.0.0.0:11434',) ## llama 3.1
    vectorstore = Chroma(persist_directory=persist_directory,
                        embedding_function=embeddings_model
                        )
    
    retriever = vectorstore.as_retriever(search_kwargs={'k': 10})

    return retriever

def get_retriever2(question, persist_directory="docs_chromadb"):
    '''Get retriever object from Chroma

    Args:
        question (str): question of RAG
        persist_directory (str): name of database
    '''
    
    model = mistral7b_llm()

    vectorstore = Chroma(persist_directory=persist_directory,
                        embedding_function=model
                        )
    
    retriever = SelfQueryRetriever.from_llm(
            model, vectorstore, question, metadata_field_info, verbose=True
        )

    return retriever