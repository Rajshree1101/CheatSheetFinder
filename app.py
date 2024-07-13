import json
import os
import sys
import boto3
import streamlit as st

## We will be using Titan Embeddings Model to generate Embedding

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader

# Vector Embedding And Vector Store

from langchain.vectorstores import FAISS

## LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

## Data ingestion
def data_ingestion(uploaded_file):
    if uploaded_file is not None:
        with open(os.path.join("data", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(os.path.join("data", uploaded_file.name))
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        docs = text_splitter.split_documents(documents)
        return docs
    return []

## Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    llm = Bedrock(model_id="ai21.j2-mid-v1", client=bedrock, model_kwargs={'maxTokens': 512})
    return llm

def get_llama2_llm():
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end but use at least summarize with 250 words with detailed explanations. If you don't know the answer, try to make up an answer.
<context>
{context}
</context>
Question: {question}
Assistant:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")
    
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            docs = data_ingestion(uploaded_file)
            if docs:
                get_vector_store(docs)
                st.success("Vector store updated with uploaded PDF.")
            else:
                st.error("Failed to process the uploaded PDF.")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings,allow_dangerous_deserialization=True)
            llm = get_claude_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings,allow_dangerous_deserialization=True)
            llm = get_llama2_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    main()
