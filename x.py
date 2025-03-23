from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_groq import ChatGroq
from fastapi.middleware.cors import CORSMiddleware
import pickle
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv
load_dotenv()
from agno.agent import Agent,RunResponse
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from phi.tools.yfinance import YFinanceTools
# FastAPI app
app = FastAPI()

# Request model
class QueryRequest(BaseModel):
    query: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

MONGO_URI = os.getenv("MONGO_URI")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
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

# Initialize LLM
llm = ChatGroq(model_name="qwen-2.5-32b", api_key=GROQ_API_KEY)

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
        print(urls)
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


analyse_pdf = f"""
XYZ FinTech Solutions Inc. – Key Compliance Data (FY 2023-24)
KYC Verification Rate: 72% → Standard: 100%.

AML Transaction Monitoring: 85% monitored monthly → Standard: 100% real-time monitoring.

Credit Information Reporting: Reported quarterly → Standard: Monthly reporting.

Loan-to-Value (LTV) Ratio: 65% → RBI cap: ≤ 75%.

NPA (Non-Performing Assets): 4.8% → RBI threshold: ≤ 4%.

AIF Due Diligence: 65% of investors reviewed annually → Standard: 100%.

Financial Disclosure Timeliness: 20-day delay → Standard: On-time disclosure.

Large Investor Verification: 50% thorough verification → Standard: 100%.

Penalty Frequency: 3 penalties in 6 months → Standard: ≤ 1 annually.

Capital Adequacy Ratio: 12.5% → SEBI threshold: ≥ 15%.

GST Filing Timeliness: 88% filed on time → Standard: 100%.

TDS Compliance Rate: 92% compliant → Standard: 100%.

Tax Audit Findings: 4 irregularities in the last quarter → Standard: ≤ 2 annually.

Tax Payment Delays: 15-20 day delays → Standard: No delays allowed.

Tax Penalties: ₹3.2 lakh in fines in the last year → Standard: No fines allowed.
"""
use_pdf = f"""Ensuring compliance with regulatory authorities such as the Reserve Bank of India (RBI), Securities and Exchange Board of India (SEBI), and the Income Tax Department is crucial for companies operating in India. Adherence to the latest guidelines helps avoid penalties and maintain operational integrity. Below are general policies and guidelines updated to reflect current compliance criteria:

1. Reserve Bank of India (RBI):

Know Your Customer (KYC) and Anti-Money Laundering (AML) Standards: Companies must implement robust KYC procedures to verify customer identities and monitor transactions to prevent money laundering. Non-compliance can result in significant penalties. For instance, the RBI imposed a ₹70 lakh penalty on a bank for non-compliance with KYC norms. ​
Wikipedia

Reporting to Credit Information Companies: Asset Reconstruction Companies (ARCs) are required to standardize their credit bureau reporting to ensure accurate reflection of borrowers' credit histories. This harmonization aids in consistent credit reporting across financial entities. ​
Reuters

2. Securities and Exchange Board of India (SEBI):

Due Diligence for Alternative Investment Funds (AIFs): Fund managers must exercise stringent due diligence to prevent circumvention of regulations, including practices like "evergreening" stressed loans. SEBI mandates a thorough review of large investors and requires government approval for investors from countries sharing land borders with India. ​
Reuters
+1
Regulatory Compliance Software
+1

Compliance with ICDR Regulations and NDI Rules: AIFs are required to adhere to Issue of Capital and Disclosure Requirements (ICDR) and Non-Debt Instruments (NDI) Rules to ensure transparency and legal adherence. SEBI's guidelines emphasize detailed due diligence on AIFs, their managers, and investors. ​
Regulatory Compliance Software
+1
Reuters
+1

3. Income Tax Department:

Timely Tax Filings and Payments: Companies must adhere to Goods and Services Tax (GST), income tax, and Tax Deducted at Source (TDS) regulations. Timely and accurate tax filings and payments are essential to avoid penalties. ​
Probe42 Blog

Compliance with Retrospective Amendments: The Central Board of Direct Taxes (CBDT) has issued guidelines for availing benefits due to retrospective amendments in tax laws. Companies should stay updated on such changes to ensure compliance and avoid disputes. ​
saspartners

4. General Compliance Measures:

Annual Filing Compliance: Filing annual returns (Form MGT-7), financial statements (Form AOC-4), and other statutory documents is essential for compliance with the Ministry of Corporate Affairs (MCA). ​
Probe42 Blog

Labour Law Compliance: Adherence to labor regulations, such as Provident Fund (PF), Employee State Insurance (ESI), and gratuity, is crucial to protect employee rights and avoid legal disputes. ​
"""

load_dotenv()
analysis_prompt = f"""
    {analyse_pdf} for this:
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
regulatory_prompt = f"""
{use_pdf} compare with reference to this for {analyse_pdf}:
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
final_output = f"""An advanced autonomous compliance management system that ensures organizations adhere to central authority regulations by systematically auditing and evaluating their policies.

Score: [Final Compliance Score (0-100)]

The system provides a structured framework of compliance criteria and guidelines, helping organizations assess their policies ({analysis_prompt}).

It verifies company policies against regulatory standards ({regulatory_prompt}) to detect violations, suggest improvements, and highlight key updates.

It assigns a final compliance score (0-100) at the start, summarizing the overall adherence level.

The system classifies compliance levels for each regulatory body (RBI, SEBI, IRDAI, Income Tax Dept., etc.) and quantifies how much (%) compliance is achieved within the 0-100 range for each authority.

It provides a detailed justification for the score, highlighting key points, necessary improvements, and risk areas that require attention.

This ensures businesses maintain transparency, mitigate risks, and stay ahead in regulatory adherence with minimal manual intervention, while offering a quantifiable and structured compliance assessment.

"""
class QueryRequest(BaseModel):
    question: str
class Query(BaseModel):
    q: str
calc = Agent(
    model=Groq(id="deepseek-r1-distill-llama-70b",temperature=0.4),
    description=f"""follow {final_output} and give numbers , labels and their weak and strong points""",
    tools=[DuckDuckGoTools(),YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    markdown=True
)

agent = Agent(
    model=Groq(id="qwen-2.5-32b",temperature=0.6),
    description=f""""An advanced autonomous compliance management system that ensures organizations adhere to central authority regulations by systematically evaluating their policies.
{use_pdf} compare with reference to this for {analyse_pdf}:
It provides a structured framework of compliance criteria and guidelines, helping organizations assess their policies ({analysis_prompt}).

It verifies company policies against regulatory standards ({regulatory_prompt}) to detect violations, suggest improvements, and highlight key updates.

The system generates actionable insights, key takeaways, and summaries to enhance compliance strategies and align company policies with evolving regulatory requirements.

This ensures businesses maintain transparency, mitigate risks, and stay ahead in regulatory adherence with minimal manual intervention.

- give complete detailed audit report of the company for 2 pages.
""",
    tools=[DuckDuckGoTools(),YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    
    markdown=True
)
qna = Agent(
    model=Groq(id="qwen-2.5-32b",temperature=0.6),
    description=f"""you know everything and also give market advice based on {use_pdf} and {analyse_pdf} and the web/internet data and help user whatever question you are asked use the data from the pdf and the web to answer the question. and give the strong source if you use web search and evry time think from different perpactive if the input is same means variate the score by 1 or 3 gap.
""",
    tools=[DuckDuckGoTools(),YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    
    markdown=True
)

# API endpoint for querying the AI agents
@app.post("/query")
async def query_agents():
    question = f"give audit and compliance for {analyse_pdf} with respact to {use_pdf} and follow the provided instructions"
    response: RunResponse = agent.run(question)
    score: RunResponse = calc.run("Give the compilance score at the start and fixed label for all audit and also which compliance is majorly not matching with company policies and regulations that also give.")
    return {"audit":  response.content, "score": score.content}
@app.post("/ans")
async def query_agents(request: Query):
    resp: RunResponse = qna.run(request.q)
   
    return {"ans":  resp.content}

# Run FastAPI server with: uvicorn script_name:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run('x:app', host="127.0.0.1", port=6700,reload=True)