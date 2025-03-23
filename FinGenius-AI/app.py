import os
import pickle
from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_groq import ChatGroq
from fastapi.middleware.cors import CORSMiddleware

# from langchain_ollama import OllamaLLM
# llm = OllamaLLM(model="gemma:2b")

# from embed import process_urls_and_create_pkl
from dotenv import load_dotenv
load_dotenv()
import os
import pickle
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def process_urls_and_create_pkl(urls, file_path="docs.pkl"):
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
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

MONGO_URI = os.getenv("MONGO_URI")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(model_name="qwen-2.5-32b", api_key=GROQ_API_KEY)
class QueryRequest(BaseModel):
    question: str
class urlreq(BaseModel):
    urls: str
url = []
@app.get("/fetch-urls")
def fetch_urls():
    try:
        # print(os.getenv("MONGO_URI"))   
        client = MongoClient(MONGO_URI)
        db = client["test"]
        collection = db["files"]
        urls = [doc["url"] for doc in collection.find({"url": {"$exists": True}}, {"url": 1})]
        client.close()
        print(str(urls[0]))
        process_urls_and_create_pkl(urls=str(urls[0]))
        return {"message": "Embedding stored successfully!"}    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# print(url)
@app.post("/embed")
def embed_data(req: urlreq):
    print(req.urls)
    try:
        process_urls_and_create_pkl(req.urls,"auth.pkl")
        return {"message": "Embedding stored successfully!","urls": url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def process_query(request: QueryRequest):
    """Query the LLM with a general question"""
    try:
        vectorstore = FAISS.load_local("./docs.pkl", HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

        result = chain.invoke({"question": request.question})
        return {"answer": result["answer"], "sources": result.get("sources", "")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ **GET Endpoint: Retrieve Processed Answer**
@app.get("/get-answer")
def get_answer(question: str):
    try:
        vectorstore = FAISS.load_local("./final.pkl", HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

        result = chain.invoke({"question": question})
        return {"answer": result["answer"], "sources": result.get("sources", "")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ **Company Report Endpoint**
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai

import os
from dotenv import load_dotenv
load_dotenv()
# openai.api_key=os.getenv("OPENAI_API_KEY")

## web search agent
web_search_agent=Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Alway include sources"],
    show_tools_calls=True,
    markdown=True,

)

## Financial agent
finance_agent=Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,

)

multi_ai_agent=Agent(
    team=[web_search_agent,finance_agent],
    instructions=["Always include sources","Use table to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# multi_ai_agent.print_response("Summarize analyst recommendation and share the latest news for NVDA",stream=True)

@app.post("/company-report")
def company_report():
    """Generate a report for a company"""
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

    #### **8. Provide all raw info about company policies and standards**
    - Details of all company policies and standards mentioned in the document.

    #### **9. Give the detailed summary**
    - Detailed summary of the document.
     
    #### **10. Provide a detailed breakdown without making comparisons or judgments.**

    ### **11. Also provide a Detailed SWOT Analysis of the company.**
    - Strengths, Weaknesses, Opportunities, and Threats of the company.
    - Pros and Cons of the company.
    - Opportunities and Threats faced by the company.

    **Provide a detailed breakdown without making comparisons or judgments. The extracted details will be further analyzed by another model.**
    """
    try:
        # Step 1: Load the FAISS vector store from the .pkl file
        try:
            vectorstore = FAISS.load_local("docs.pkl", HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading vector store: {str(e)}")

        # Step 2: Set up the retriever for querying the vector store
        try:
            retriever = vectorstore.as_retriever()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error setting up retriever: {str(e)}")

        # Step 3: Initialize the retrieval chain with the LLM and the retriever
        try:
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error initializing retrieval chain: {str(e)}")

        # Step 4: Execute the chain with the provided analysis prompt
        try:
            result = chain.run({"question": analysis_prompt})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error executing the retrieval chain: {str(e)}")

        # Step 5: Return the answer and sources, handling missing data
        try:
            return {
                "company_report": result.get("answer", "No data found"),
                "sources": result.get("sources", "No sources found")
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving the result: {str(e)}")

    except Exception as e:
        # Catch any unhandled exceptions
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")# ✅ **Authority Report Endpoint**
@app.post("/authority-report")
def authority_report(request: QueryRequest):
    """Generate a report for an authority"""
    regulatory_prompt = """
Extract and summarize all key compliance-related information from the provided official regulations, guidelines, and standards issued by centralized regulatory authorities or institutions.

The output should be structured, with each point summarized in 3 to 4 lines, maintaining the following format:

### **Regulatory Compliance Standards & Guidelines**

#### **1. Overview of Regulatory Framework**
- List the key laws, guidelines, and regulatory standards issued by the authority.

#### **2. Financial & Tax Compliance Regulations**
- Summarize legal obligations regarding financial disclosures, taxation policies, and financial reporting.

#### **3. Legal & Regulatory Obligations**
- Detail the mandatory legal requirements that organizations must comply with under this regulation.

#### **4. Data Privacy & Cybersecurity Regulations**
- Summarize key rules on data protection, cybersecurity, GDPR-like compliance, and privacy laws.

#### **5. Industry-Specific Regulations & Compliance**
- Extract compliance measures specific to industries (e.g., healthcare, finance, defense, etc.).

#### **6. Risk Management & Internal Control Requirements**
- Outline prescribed methods for risk assessment, internal audits, and governance frameworks.

#### **7. Ethical Standards & Employee Compliance Requirements**
- Summarize compliance expectations related to corporate ethics, employee conduct, and whistleblower protection.

#### **8. Comprehensive List of Policy & Standard Requirements**
- Provide details of all regulatory policies and standards that must be followed.

#### **9. Detailed Summary of the Regulatory Framework**
- Generate a structured summary of the entire regulation or guideline.

#### **10. In-Depth Compliance Breakdown**
- Offer a detailed, neutral analysis of each regulatory requirement, ensuring clarity and completeness.

#### **11. SWOT Analysis of Regulatory Impact**
- Provide a SWOT (Strengths, Weaknesses, Opportunities, and Threats) analysis of the regulatory framework itself:
  - **Strengths:** Benefits of the regulation.
  - **Weaknesses:** Challenges or gaps in implementation.
  - **Opportunities:** How organizations can leverage compliance for advantages.
  - **Threats:** Potential risks, penalties, or enforcement challenges.

Ensure that the output provides a neutral and structured breakdown, without making subjective judgments. This extracted data will be used as a benchmark to compare with individual company compliance reports.
"""

    try:
        vectorstore = FAISS.load_local("./authority.pkl", HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

        result = chain.invoke({"question": f"{regulatory_prompt}"})
        return {"authority_report": result.get("answer", "No data found"), "sources": result.get("sources", "No sources found")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ **Agent Report Endpoint**
@app.post("/agent-report")
def agent_report(request: QueryRequest):
    """Generate a report for an agent"""
    try:
        vectorstore = FAISS.load_local("./final.pkl", HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

        result = chain.invoke({"question": f"Generate a detailed agent report for: {request.question}"})
        return {"agent_report": result.get("answer", "No data found"), "sources": result.get("sources", "No sources found")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run('main:app', host="127.0.0.1", port=7000,reload=True)