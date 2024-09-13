'''FAST API to serve REST framework for analyzing real estate listings
'''
import os
import sys
from typing import Optional, Literal
from pydantic import BaseModel

from fastapi import FastAPI
import uvicorn

from persona_private.ai.chroma_client import get_retriever
from persona_private.ai.agent import rag, mistral7b_llm
from persona_private.ai.chroma_client import load_docs, load_json, doc2chroma, save_docs2_chroma, get_retriever

import dotenv

dotenv.load_dotenv()


llm = mistral7b_llm()

app = FastAPI()

class QueryParams(BaseModel):
    question: str

class JsonFile(BaseModel):
    file: dict
    filename: str

def trigger_reload():
    """Function to trigger the reload process by restarting the server."""
    print("sys.argv>>", sys.argv)
    print(__file__)
    os.execv(__file__, ['python'] + sys.argv)

@app.get('/')
async def home():
    return {"message": "RAG AI Private Persona FAST API is running!!"}

@app.post('/analyze')
def process(query_params: QueryParams):
    trigger_reload()
    retriever = get_retriever(persist_directory="docs_chromadb")
    try:
        question = query_params.model_dump()['question']
        answer = rag(retriever, llm, question=question)
        return {"answer": answer}
    except Exception as e:
        print('Error>>>', e)
        return 500, {"message": f"Following error occured>> {e}"}

@app.post('/embeed')
def embeed(json_file: JsonFile):
    print("json_file>>", json_file)
    # try:
    json_obj = json_file.model_dump()['file']
    filename = json_file.model_dump()['filename']
    print('json_file>>', json_file)

    docs = load_json(json_obj=json_obj, filename=filename)

    retriever = doc2chroma(docs=docs, persist_directory="docs_chromadb")
    print("retriever>>", retriever)
    return {"message": f"{filename} was save to Chroma!!!"}
    # except Exception as e:
    #     print('Error>>>', e)
    #     return 500, {"message": f"Following error occured>> {e}"}
    


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)