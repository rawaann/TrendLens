from langchain.agents import initialize_agent, AgentType, Tool
from tools.verification_tools import verify_information_tool
from config import llm

def create_verification_agent():
    verify_tools = [
        Tool(
            name="verify_information",
            func=verify_information_tool,
            description="Verify claims against paper content",
            return_direct=True
        )
    ]
    return initialize_agent(
        tools=verify_tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=False,
        system_message=(
            "Do NOT use inner thoughts or chain-of-thought reasoning. ONLY use your tool directly and return the result. "
            "You are an academic fact-checker. Your ONLY job is to verify claims against paper content. "
            "NEVER try to summarize, analyze trends, or answer general questions - just verify factual accuracy. "
            "You MUST ONLY use the tools provided to you and NEVER answer from your own knowledge or without using a tool. "
            "If you cannot answer using your tools, respond: 'I can only answer using my tools and cannot provide an answer otherwise.'"
        )
    ) 