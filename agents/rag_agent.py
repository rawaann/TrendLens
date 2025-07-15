from langchain.agents import initialize_agent, AgentType, Tool
from tools.rag_tools import rag_tool
from config import llm

def create_rag_agent():
    rag_tools = [
        Tool(
            name="rag",
            func=rag_tool,
            description="ONLY answer specific, direct questions about the content of the papers. NOT for summaries or overviews. Example input: 'What method does the first paper use?' Example output: [answer from rag tool]",
            return_direct=True
        )
    ]
    return initialize_agent(
        tools=rag_tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=False,
        system_message=(
            "You are a research QA specialist. For every question, think step by step and use inner thoughts (chain-of-thought reasoning) before answering. NEVER use this tool for summarizing, overviews, or main points of a paper. ONLY use for direct Q&A about specific details. "
            "Your ONLY job is to answer specific, direct questions about academic papers using their content through RAG. "
            "NEVER use this tool for summaries, overviews, or general descriptions. ONLY use for direct Q&A. "
            "You MUST ONLY use the tools provided to you and NEVER answer from your own knowledge or without using a tool. "
            "If you cannot answer using your tools, respond: 'I can only answer using my tools and cannot provide an answer otherwise.' "
            "Example input: 'What method does the first paper use?'\n"
            "Example output: [answer from rag tool]"
        )
    ) 