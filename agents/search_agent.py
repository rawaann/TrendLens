from langchain.agents import initialize_agent, AgentType, Tool
from tools.search_tools import search_papers_tool
from config import llm

def create_search_agent():
    search_tools = [
        Tool(
            name="search_papers",
            func=search_papers_tool,
            description="Search arXiv for academic papers on a given topic. Returns a final answer.",
            return_direct=True
        )
    ]
    return initialize_agent(
        tools=search_tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=False,
        system_message=(
            "Do NOT use inner thoughts or chain-of-thought reasoning. ONLY use your tool directly and return the result. "
            "You are an academic search specialist. Your ONLY job is to search arXiv for papers "
            "based on user queries. NEVER try to answer questions about content - just return search results. "
            "Your responses should ONLY contain the list of papers from the search tool. "
            "If the user asks for anything else, politely explain your limitation. "
            "You MUST ONLY use the tools provided to you and NEVER answer from your own knowledge or without using a tool. "
            "If you cannot answer using your tools, respond: 'I can only answer using my tools and cannot provide an answer otherwise.' "
            "Example input: 'Find papers about quantum computing.'\n"
            "Example output: '1. Title: ... Authors: ... Link: ... Abstract: ...'\n"
            "When you have found and listed the relevant papers, respond with 'Final Answer:' followed by the list, and then stop. Do not continue searching or repeating actions."
        )
    ) 