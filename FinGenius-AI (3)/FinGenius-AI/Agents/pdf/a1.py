import os
import streamlit as st
import pickle
import time
from gtts import gTTS
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI


from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.title("FinGenius📈")
st.sidebar.title("News Article URLs")

from pymongo import MongoClient

# Static URLs
# urls = [
#     "https://www.moneycontrol.com/news/business/tata-motors-mahindra-gain-certificates-for-production-linked-payouts-11281691.html",
    
# ]
MONGO_URI = os.getenv("MONGO_URI")

# Connect to MongoDB and retrieve URLs
def fetch_urls_from_mongodb():
    client = MongoClient(MONGO_URI)
    db = client["test"]  # Database name
    collection = db["files"]  # Collection name
    urls = [doc["url"] for doc in collection.find({"url": {"$exists": True}}, {"url": 1})]
    client.close()
    return urls

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "docs.pkl"

main_placeholder = st.empty()
# llm = ChatGroq(model_name="qwen-2.5-32b", api_key=GROQ_API_KEY)
llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
# llm = ChatGoogleGenerativeAI(model="gemma-2-2b-it")
# from langchain_ollama import OllamaLLM
# llm = OllamaLLM(model="gemma:2b")

if process_url_clicked:
    # try:
    #     loader = PyPDFLoader("https://res.cloudinary.com/dkuitm79x/image/upload/v1742575466/pub-ch-compliance-management-systems.pdf.pdf")
    #     main_placeholder.text("Data Loading...Started...✅✅✅")
    #     data = loader.load()
    # except Exception:
    #     st.text("OnlinePDFLoader failed, switching to UnstructuredURLLoader...🚀")
    #     loader = UnstructuredURLLoader(urls = "https://res.cloudinary.com/dkuitm79x/image/upload/v1742575466/pub-ch-compliance-management-systems.pdf.pdf")
    #     data = loader.load()

    # text_splitter = RecursiveCharacterTextSplitter(
    #     separators=['\n\n', '\n', '.', ','],
    #     chunk_size=1000
    # )
    # main_placeholder.text("Text Splitter...Started...✅✅✅")
    # docs = text_splitter.split_documents(data)
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # new_vectorstore = FAISS.from_documents(docs, embeddings)
    # main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    # time.sleep(2)

#     if os.path.exists(file_path):
#         with open(file_path, "rb") as f:
#             old_vectorstore = pickle.load(f)
#         new_vectorstore.merge_from(old_vectorstore)  

#     with open(file_path, "wb") as f:
#         pickle.dump(new_vectorstore, f)
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

# query = main_placeholder.text_input("Question: ")
# if query:
if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": analysis_prompt}, return_only_outputs=True)
        
        st.header("Answer")
        report_text = result["answer"]
        st.write(report_text)

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
    query = st.text_input("Question: ")

    if query:
            # Perform Q&A with the data source
            result = chain({"question": query}, return_only_outputs=True)
            
            # Display the answer
            st.header("Answer")
            report_text = result["answer"]
            st.write(report_text)

            # Display the sources if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                for source in sources.split("\n"):
                    st.write(source)