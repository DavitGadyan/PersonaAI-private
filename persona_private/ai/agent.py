'''
AI RAG Agent
'''
import os
import torch
import nest_asyncio
import transformers
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from transformers import (
  AutoTokenizer, 
  AutoModelForCausalLM, 
  BitsAndBytesConfig,
  pipeline
)
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import login
login(token=os.environ["SECRET_HF"])

def mistral7b_llm():
    llm = Ollama(  model="mistral",
                    #model='Llama2',
                    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]))
    
    
    return llm

# def mistral7b_llm():
#     '''Load LLM Mistral 7B
#     '''
#     #################################################################
#     # Tokenizer
#     #################################################################

#     model_name='mistralai/Mistral-7B-Instruct-v0.1'

#     model_config = transformers.AutoConfig.from_pretrained(
#         model_name,
#     )

#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=os.environ["SECRET_HF"])
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "right"

#     #################################################################
#     # bitsandbytes parameters
#     #################################################################

#     # Activate 4-bit precision base model loading
#     use_4bit = True

#     # Compute dtype for 4-bit base models
#     bnb_4bit_compute_dtype = "float16"

#     # Quantization type (fp4 or nf4)
#     bnb_4bit_quant_type = "nf4"

#     # Activate nested quantization for 4-bit base models (double quantization)
#     use_nested_quant = False

#     #################################################################
#     # Set up quantization config
#     #################################################################
#     compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=use_4bit,
#         bnb_4bit_quant_type=bnb_4bit_quant_type,
#         bnb_4bit_compute_dtype=compute_dtype,
#         bnb_4bit_use_double_quant=use_nested_quant,
        
#     )

#     # Check GPU compatibility with bfloat16
#     if compute_dtype == torch.float16 and use_4bit:
#         major, _ = torch.cuda.get_device_capability()
#         if major >= 8:
#             print("=" * 80)
#             print("Your GPU supports bfloat16: accelerate training with bf16=True")
#             print("=" * 80)

#     #################################################################
#     # Load pre-trained config
#     #################################################################
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         token=os.environ["SECRET_HF"],
#         quantization_config=bnb_config,
#     )

#     text_generation_pipeline = transformers.pipeline(
#         model=model,
#         tokenizer=tokenizer,
#         task="text-generation",
#         temperature=0.2,
#         repetition_penalty=1.1,
#         return_full_text=True,
#         max_new_tokens=300,
#     )

#     prompt_template = """
#     ### [INST] 
#     Instruction: Answer the question based on your knowledge. Here is context to help:

#     {context}

#     ### QUESTION:
#     {question} 

#     [/INST]
#     """

#     mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

#     # Create prompt from prompt template 
#     prompt = PromptTemplate(
#         input_variables=["context", "question"],
#         template=prompt_template,
#     )

#     # Create llm chain 
#     llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

#     return llm_chain


def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    # for source in llm_response["source_documents"]:
    #     print("sourceXXX>>>", source)
    #     print(source.metadata['source'])
    
    return {
        "answer": llm_response['result'],
        # "source_documents": llm_response["source_documents"]

    }

def rag(retriever, llm, question):
    '''RAG Agent to answer questions

    Args:
        retriever (Pinecone.Vectorstore.Retriever): Retriever
        llm (LLM): llm model
        question (str): question string
    
    '''
    # rag_chain = ( 
    # {"context": retriever, "question": RunnablePassthrough()}
    #     | llm )
    # out = rag_chain.invoke(question)

    # ## separate output
    # output_answer = out["text"].split("[/INST]")[-1].strip()


    # Question
    # qa_chain = RetrievalQA.from_chain_type(llm=llm,
    #                                 chain_type="stuff",
    #                                 retriever=retriever,
    #                                 return_source_documents=True,
    #                                 verbose=True)
    
    # llm_response = qa_chain(question)
    llm_response = retriever.invoke(question)
    print("llm_response>>", llm_response)
    print("llm_response>>", llm_response.keys())
    print("source docs len >>", len(llm_response["source_documents"]))
    print("source docs >>", llm_response["source_documents"])
    output_answer = process_llm_response(llm_response)



    return output_answer["answer"]