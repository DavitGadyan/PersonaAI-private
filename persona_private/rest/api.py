'''FAST API to serve REST framework for analyzing real estate listings
'''
from typing import Optional, Literal
from pydantic import BaseModel

from fastapi import FastAPI
import uvicorn

from persona_private.ai.chroma_client import get_retriever
from persona_private.ai.agent import rag, mistral7b_llm
from persona_private.ai.chroma_client import save_docs2_chroma
import dotenv

dotenv.load_dotenv()

retriever = get_retriever(persist_directory="docs_chromadb")
llm = mistral7b_llm()

app = FastAPI()

class DocParams(BaseModel):
    question: str

@app.get('/')
async def home():
    return {"message": "RAG AI Private Persona FAST API is running!!"}

@app.post('/analyze')
def process(doc_params: DocParams):
    try:
        question = doc_params.model_dump()['question']
        answer = rag(retriever, llm, question=question)
        return {"answer": answer}
    except Exception as e:
        print('Error>>>', e)
        return 500, {"message": f"Following error occured>> {e}"}

    return {"message": f"{doc_params.dict()} has been processed!!!"}


@app.post('/embeed')
def process(docs_path: str):
    # try:
    print('docs_path>>', docs_path)

    try:
        retriever = save_docs2_chroma(docs_path)
    except:
        retriever = save_docs2_chroma(docs_path)
        
    # except Exception as e:
    #     print('Error>>>', e)
    #     return 500, {"message": f"Following error occured>> {e}"}
    
    return {"message": f"{doc_obj.keys()} has been saved to Pinecone!!!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)