'''FAST API to serve REST framework for analyzing real estate listings
'''
import os
import sys
import time
from typing import Optional, Literal
from pydantic import BaseModel

from fastapi import FastAPI
import uvicorn

from persona_private.ai.chroma_client import get_retriever, get_retriever2
from persona_private.ai.agent import rag, mistral7b_llm
from persona_private.ai.chroma_client import load_docs, load_json, doc2chroma, save_docs2_chroma, get_retriever, load_json2, get_retriever3

import dotenv

dotenv.load_dotenv()


llm = mistral7b_llm()

app = FastAPI()

class QueryParams(BaseModel):
    question: str

class JsonFile(BaseModel):
    file: dict
    filename: str



@app.get('/')
async def home():
    return {"message": "RAG AI Private Persona FAST API is running!!"}

@app.post('/analyze')
def process(query_params: QueryParams):
    # try:
    question = query_params.model_dump()['question']
    retriever = get_retriever3(question=question, persist_directory="docs_chromadb")
    docs = retriever.get_relevant_documents(question)
    print(len(docs))
    print(docs)
    time.sleep(10)
    answer = rag(retriever, llm, question=question)
    return {"answer": answer}
    # except Exception as e:
    #     print('Error>>>', e)
    #     return 500, {"message": f"Following error occured>> {e}"}

@app.post('/embeed')
def embeed(json_file: JsonFile):
    # print("json_file>>", json_file)
    # try:
    json_obj = json_file.model_dump()['file']
    filename = json_file.model_dump()['filename']
    # print('json_file>>', json_file)

    docs = load_json2(json_obj=json_obj, filename=filename)

    retriever = doc2chroma(docs=docs, persist_directory="docs_chromadb")
    print("retriever>>", retriever)
    return {"message": f"{filename} was save to Chroma!!!"}
    # except Exception as e:
    #     print('Error>>>', e)
    #     return 500, {"message": f"Following error occured>> {e}"}
    


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000,
    
    reload_includes=["docs_chromadb/*.sqlite3"])