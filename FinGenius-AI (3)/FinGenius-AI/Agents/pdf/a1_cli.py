import os
import pickle
import time
from pymongo import MongoClient
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
file_path = "docs.pkl"

llm = ChatGroq(model_name="qwen-2.5-32b", api_key=GROQ_API_KEY)
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")


def fetch_urls_from_mongodb():
    """Fetches URLs from MongoDB."""
    client = MongoClient(MONGO_URI)
    db = client["test"]  # Database name
    collection = db["files"]  # Collection name
    urls = [doc["url"] for doc in collection.find({"url": {"$exists": True}}, {"url": 1})]
    client.close()
    return urls

def process_documents():
    """Loads and processes PDFs or web documents into a vector database."""
    try:
        loader = PyPDFLoader("https://res.cloudinary.com/dkuitm79x/image/upload/v1742575466/pub-ch-compliance-management-systems.pdf.pdf")
        data = loader.load()
    except Exception:
        loader = UnstructuredURLLoader("https://www.moneycontrol.com/news")
        data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_vectorstore = FAISS.from_documents(docs, embeddings)

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            old_vectorstore = pickle.load(f)
        new_vectorstore.merge_from(old_vectorstore)  

    with open(file_path, "wb") as f:
        pickle.dump(new_vectorstore, f)

    return new_vectorstore

def generate_compliance_report():
    """Generates a compliance report based on extracted document information."""
    analysis_prompt = """
    Give and Extract and summarize all key compliance-related information from the provided company compliance documents.
    The output should be structured and each point answer should be in 3 to 4 lines in the following format based on whatever data you provided doesn't matter its sufficient or not.:
     

    ### **Company Compliance Report Analysis**
    #### **1. Overview of Compliance Policies**
    - List the compliance policies and standards mentioned in the document.

    #### **2. Financial & Tax Compliance Details**
    - Summarize details related to financial disclosures, tax policies, and financial reporting.

    #### **3. Legal & Regulatory Compliance**
    - List key legal obligations and regulations the company follows.

    #### **4. Data Privacy & Security Policies**
    - Summarize company policies related to data protection, cybersecurity, and GDPR-like compliance.

    #### **5. Industry-Specific Compliance**
    - Extract any specific compliance measures relevant to the company's industry.

    #### **6. Risk Management & Internal Controls**
    - Summarize how the company handles risk management and internal audits.

    #### **7. Employee & Ethical Compliance**
    - List company policies on ethical behavior, whistleblower protection, and employee compliance.

    #### **8. provide all raw info about company policies and standards.
    - details of all company policies and standards mentioned in the document.

    #### **9. give the detailed summary.
    - detailed summary of the document.
     
    #### **10. Provide a detailed breakdown without making comparisons or judgments.

    ### **11. also Provide a Detailed SWOT Analysis of the company.
    - Strengths, Weaknesses, Opportunities, and Threats of the company.
    - Pros and Cons of the company.
    - Opportunities and Threats faced by the company.

    **Provide a detailed breakdown without making comparisons or judgments. The extracted details will be further analyzed by another model.**
    
    """


    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain.invoke({"question": analysis_prompt}, return_only_outputs=True)
        return result["answer"]
    
    return "No document data available."

# Example usage
if __name__ == "__main__":
    process_documents()
    report = generate_compliance_report()
    print(report)
