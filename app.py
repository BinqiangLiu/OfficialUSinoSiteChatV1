import streamlit as st
#from langchain.document_loaders import UnstructuredURLLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from langchain import HuggingFaceHub
import os
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from langchain.chat_models import ChatOpenAI
#import nltk
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai_api_key = os.environ.get('openai_api_key')
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

pinecone_index_name=os.environ.get('pinecone_index_name')
pinecone_namespace=os.environ.get('pinecone_namespace')
pinecone_api_key=os.environ.get('pinecone_api_key')
pinecone_environment=os.environ.get('pinecone_environment')
pinecone_index_name=os.environ.get('pinecone_index_name')
pinecone.init(      
	api_key=pinecone_api_key,      
	environment=pinecone_environment      
)      
index = pinecone.Index(pinecone_index_name)
loaded_v_db_500_wt_metadata = Pinecone.from_existing_index(index_name=pinecone_index_name, embedding=embeddings, namespace=pinecone_namespace)

hf_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
HUGGINGFACEHUB_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
huggingfacehub_api_token= os.environ.get('huggingfacehub_api_token')
repo_id = os.environ.get('repo_id')

prompt_template = """
#You are a very helpful AI assistant. Please ONLY use {context} to answer the user's input question. If you don't know the answer, just say that you don't know. DON'T try to make up an answer and do NOT go beyond the given context without the user's explicitly asking you to do so!
You are a very helpful AI assistant. Please response to the user's input question with as many details as possible.
Question: {question}
Helpful AI Repsonse:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

llm = HuggingFaceHub(repo_id=repo_id,
                     #huggingfacehub_api_token="hf_p***K",
                     huggingfacehub_api_token=hf_token,
                     model_kwargs={"min_length":1024,
                                   "max_new_tokens":5632, "do_sample":True,
                                   "temperature":0.1,
                                   "top_k":50,
                                   "top_p":0.95, "eos_token_id":49155}) 


user_query=st.text_input("Enter your query:")
memory_NEW = ConversationBufferMemory(memory_key="chat_history", return_messages= True)
loaded_doc_retriever = loaded_v_db_500_wt_metadata.as_retriever(search_type="similarity", search_kwargs={"k":2})
new_ConverRtr_chain = ConversationalRetrievalChain.from_llm(llm, retriever=loaded_doc_retriever, memory=memory_NEW)
new_ConverRtr_chain_result=new_ConverRtr_chain({'question': user_query})
ai_response=new_ConverRtr_chain_result['answer']
st.write("AI Response:")
st.write(ai_response)




         
