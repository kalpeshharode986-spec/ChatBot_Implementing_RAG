import numpy as np
import pandas as pd
import streamlit as st 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

 # pdf document ko read karne ke liye

from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader(r"C:\Users\ASUS\Downloads\nvdoc2.pdf")
 # for importing pdf file
pages = loader.load()

full_text = ""
for i in pages:
    full_text += i.page_content

import re # regex for text cleaning
full_text = re.sub(r'[^\w\s]', '', full_text) # removing punctuation

full_text = full_text.replace('\n', ' ') # removing new line characters

from langchain.schema import Document # full_text ko document me convert krna
combined = Document(page_content=full_text, metadata={"source": "nvdoc2.pdf"})

from langchain_text_splitters import RecursiveCharacterTextSplitter # for splitting the text into chunks 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=70)
chunks = text_splitter.split_documents([combined])


def clean_text(text): # utf-8 me conversion k liye 
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")

for i in chunks:
    i.page_content = clean_text(i.page_content)

from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2",
    model_kwargs={"trust_remote_code": True} )

from langchain_community.vectorstores import FAISS
db = FAISS.from_documents(chunks, embeddings )

db.save_local("My_db") # saving the vector database

db = FAISS.load_local("My_db", embeddings , allow_dangerous_deserialization=True)

# hwllo

query = input("Enter your question: ") # user se question lena

docs = db.similarity_search(query, k=5) # top 5 similar documents ke liye

all_content = "" # sabhi documents ka content ek sath jodna
for doc in docs:
    all_content += doc.page_content + " "

llm = ChatOpenAI(
    temperature=0.3,
    model="utter-project/eurollm-9b-instruct",
    openai_api_key="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",
    openai_api_base="https://integrate.api.nvidia.com/v1")





prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant. Use the following context to answer the question . Use the following pieces of context to answer the question at the end.If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n" # ye ek prompt hai ise hum change kar sakte h 
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ))

prompt = prompt_template.format(context=all_content, question=query) # yha hum hamara data fit kar rhe h 

response = llm.invoke(prompt) # llm call karne k liye 
print("User Question:", query)
print("\nLLM Answer:", response.content)

# ye code humare rag system ka final code hai jisme humne user se question leke uska answer diya h
# ye code humare rag system ka final code hai jisme humne user se question leke uska answer diya h
