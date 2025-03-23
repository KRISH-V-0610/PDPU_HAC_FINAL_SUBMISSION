import os
import pickle
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def process_urls_and_create_pkl(urls, file_path="auth.pkl"):
    try:
        loader = PyPDFLoader(urls)
        data = loader.load()
    except Exception:
        loader = UnstructuredURLLoader(urls = urls)
        data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_vectorstore = FAISS.from_documents(docs, embeddings)
    time.sleep(2)

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            old_vectorstore = pickle.load(f)
        new_vectorstore.merge_from(old_vectorstore)

    with open(file_path, "wb") as f:
        pickle.dump(new_vectorstore, f)
    
    print("Vectorstore saved to", file_path)

# Example usage
# urls = [
#     "https://www.moneycontrol.com/news",
#     "https://www.bbc.com/news/business",
# ]
# urls = ["https://res.cloudinary.com/dkuitm79x/image/upload/v1742615242/Doc_20250322_Wa0001_2025-03-22.pdf.pdf"]
# process_urls_and_create_pkl("https://rbidocs.rbi.org.in/rdocs/notification/PDFs/NOT126F8D4D6C0D2C945718EE421C4C7711826.PDF")
