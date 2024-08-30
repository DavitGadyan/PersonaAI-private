'''
Agent testing
'''
import dotenv

dotenv.load_dotenv()
# import os
# def ensure_directory_exists(directory_path):
#     if not os.path.exists(directory_path):
#         os.makedirs(directory_path)
# ensure_directory_exists("static")     

# from persona_private.frontend import app



# if __name__ == "__main__":
#     app.run()


from persona_private.ai.chroma_client import load_docs, doc2chroma, save_docs2_chroma, get_retriever
from persona_private.ai.agent import rag, mistral7b_llm

docs = load_docs(path="persona_private/ai/files")
print("docs>>", docs)

retriever = doc2chroma(docs=docs, persist_directory="docs_chromadb")
print("retriever>>", retriever)

retriever = get_retriever(persist_directory="docs_chromadb")
print("retriever>>", retriever)

llm = mistral7b_llm()
answer = rag(retriever=retriever, llm=llm, question="What is documemt about?")

print("answer>>>", answer)