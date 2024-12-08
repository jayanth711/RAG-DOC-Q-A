import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings

from dotenv import load_dotenv
load_dotenv()

##load the groq API
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

groq_api_key= os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Gemma-7b-IT")

prompt= ChatPromptTemplate.from_template(
    """ 
    Answer the questions based on provided context only.
    please provide the most accurate response based on questio
    <context>
    {context}
    <context>
    question:{input}

    """

)

def create_vector_embedding():
       if "vectors" not in st.session_state:
              st.session_state.embeddings=OpenAIEmbeddings()
              st.session_state.loader= PyPDFDirectoryLoader("/research_papers")  ### DATA INGESTION
              st.session_state.docs= st.session_state.loader.load()  ## DOCUMENT LOADER
              st.session_state.text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
              st.session_state.final_documents= st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
              st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        
user_prompt= st.text_input("Enter your query from the research paper")

if st.button("Documents Embedding"):
       create_vector_embedding()
       st.write("vector Database is ready")

import time

if user_prompt:
    document_chain= create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retrieval_chain.invoke({'input':user_prompt})
    print(f"Response time ;{time.process_time()-start}")

    st.write(response['answer'])

    ## With streamlit expander
    with st.expander("Document similarity Search"):
          for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('--------------')
