
# from langchain_huggingface import HuggingFacePipeline
# from dotenv import load_dotenv
# import os
# load_dotenv()
# HF_TOKEN = os.getenv("HF_TOKEN")
# llm = HuggingFacePipeline.from_model_id(
#     model_id="microsoft/Phi-3-mini-4k-instruct",
#     task="text-generation",
#     pipeline_kwargs={
#         "max_new_tokens": 100,
#         "top_k": 50,
#         "temperature": 0.1,
#     },
# )
# llm.invoke("What is the capital of France?")  # Test the model
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
# llm = ChatGoogleGenerativeAI(model="gemma-2-2b-it")
print(llm.invoke("hi"))