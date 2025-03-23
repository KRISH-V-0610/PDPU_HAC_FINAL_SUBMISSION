import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools

load_dotenv()

agent = Agent(
    model=Groq(id="qwen-2.5-32b"),
    description="An intelligent web search agent that performs real-time searches to retrieve the most accurate, up-to-date, and relevant information.\n"
                "It prioritizes authoritative sources, extracts key insights, and presents structured summaries for user queries.\n"
                "The agent ensures all responses are backed by authentic sources and provides direct links for further verification.",
    tools=[DuckDuckGoTools()],
    markdown=True
)


agent.print_response("Who won the India vs Newzealand finals in CT 2025")