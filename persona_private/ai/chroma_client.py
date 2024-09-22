# Import libraries
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.embeddings import OpenAIEmbeddings
import os
import re
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
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


country_regex =  r'zimbabwe|falkland islands|antarctica|colombia|iceland|laos|timor-leste|mexico|angola|nepal|benin|algeria|spain|kosovo|united states|heard island and mcdonald islands|bolivia|montenegro|uganda|niger|cyprus|canada|cayman islands|china|martinique|liechtenstein|jamaica|bouvet island|morocco|cambodia|bahrain|grenada|uzbekistan|malaysia|sudan|cape verde|south sudan|guatemala|chile|gibraltar|pitcairn islands|tuvalu|republic of the congo|honduras|greece|trinidad and tobago|egypt|somalia|venezuela|barbados|burkina faso|belarus|zambia|saint pierre and miquelon|south korea|moldova|france|senegal|iraq|hong kong|jersey|liberia|fiji|western sahara|chad|netherlands|oman|luxembourg|turkey|taiwan|jordan|philippines|northern mariana islands|united kingdom|united arab emirates|togo|saint kitts and nevis|uruguay|mongolia|finland|portugal|anguilla|turks and caicos islands|mauritius|belgium|denmark|poland|norfolk island|ecuador|tanzania|vietnam|guam|ghana|christmas island|ukraine|niue|argentina|montserrat|gabon|qatar|djibouti|libya|norway|brunei|sweden|bhutan|el salvador|azerbaijan|myanmar|lesotho|kuwait|saint barthélemy|kiribati|south africa|lithuania|switzerland|eritrea|equatorial guinea|new zealand|iran|turkmenistan|saint helena, ascension and tristan da cunha|ethiopia|dr congo|bulgaria|north macedonia|tajikistan|kenya|malta|peru|nicaragua|dominica|south georgia|united states minor outlying islands|san marino|new caledonia|afghanistan|mauritania|saint vincent and the grenadines|rwanda|micronesia|british indian ocean territory|mayotte|guinea|latvia|french polynesia|india|solomon islands|antigua and barbuda|austria|comoros|réunion|japan|vatican city|syria|slovakia|belize|kyrgyzstan|indonesia|central african republic|pakistan|singapore|curaçao|american samoa|botswana|caribbean netherlands|mozambique|kazakhstan|germany|bangladesh|samoa|italy|mali|isle of man|gambia|romania|british virgin islands|united states virgin islands|faroe islands|svalbard and jan mayen|namibia|palau|croatia|palestine|french southern and antarctic lands|são tomé and príncipe|macau|guadeloupe|eswatini|burundi|papua new guinea|dominican republic|french guiana|nauru|armenia|puerto rico|tunisia|sierra leone|marshall islands|wallis and futuna|serbia|monaco|guinea-bissau|åland islands|saudi arabia|north korea|tonga|cook islands|haiti|russia|greenland|bermuda|guyana|ivory coast|panama|sint maarten|vanuatu|brazil|australia|cuba|costa rica|bosnia and herzegovina|albania|estonia|bahamas|saint martin|lebanon|guernsey|malawi|seychelles|georgia|tokelau|madagascar|slovenia|andorra|maldives|sri lanka|thailand|cameroon|czechia|cocos (keeling) islands|saint lucia|nigeria|aruba|suriname|yemen|hungary|ireland|paraguay'
class StringLoader:
    def __init__(self, text: str, metadata: str):
        self.text = text
        self.metadata = metadata

    def load(self):
        # Create a Document from the string
        return [Document(page_content=self.text, metadata=self.metadata)]

class CustomQueryConstructor:
    """Custom query constructor to ensure proper filter and query structure."""
    
    def __call__(self, query):
        # Correctly return the query and filter object
        return {
            "query": query,
            "filter": {
                "CountryName": "Spain"  # This could be dynamic based on user input
            }
        }


load_dotenv()

metadata_field_info = [
    AttributeInfo(
        name="CountryName",
        description="The country",
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
    metadata["CountryName"] = record.get("CountryName")
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
        data[k]["CountryName"] = k.lower()

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
    embeddings_model = OllamaEmbeddings(model="mistral", base_url='http://0.0.0.0:11434',) ## llama 3.1

    model = Ollama(model="llama3")

    vectorstore = Chroma(persist_directory=persist_directory,
                        embedding_function=embeddings_model
                        )
    document_content_description = "Indicators related to a country."

    retriever = SelfQueryRetriever.from_llm(
            model, vectorstore, document_content_description, metadata_field_info, verbose=True
        )
    # docs = retriever.get_relevant_documents(question)
    # print("docs>>", docs)
    time.sleep(10)
    return retriever

def get_retriever3(question, persist_directory="docs_chromadb"):
    '''Get retriever object from Chroma

    Args:
        question (str): question of RAG
        persist_directory (str): name of database
    '''
    embeddings_model = OllamaEmbeddings(model="mistral", base_url='http://0.0.0.0:11434',) ## llama 3.1
    vectorstore = Chroma(persist_directory=persist_directory,
                        embedding_function=embeddings_model
                        )
    country = re.findall(country_regex, question.lower())[0]
    print("country>>", country)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 10, 'filter': {'CountryName' : country}})

    return retriever