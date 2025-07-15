from langchain.agents import initialize_agent, AgentType, Tool
from agents.search_agent import create_search_agent
from agents.rag_agent import create_rag_agent
from agents.summarization_agent import create_summarization_agent
from agents.trend_agent import create_trend_agent
from agents.verification_agent import create_verification_agent
from config import llm

def create_master_agent():
    search_agent = create_search_agent()
    rag_agent = create_rag_agent()
    summary_agent = create_summarization_agent()
    trend_agent = create_trend_agent()
    verify_agent = create_verification_agent()

    routing_tools = [
        Tool(
            name="search_agent",
            func=lambda q, chat_history=None: search_agent.invoke({"input": q, "chat_history": chat_history or []})["output"],
            description="Useful for searching arXiv for papers on a topic",
            return_direct=True
        ),
        Tool(
            name="rag_agent",
            func=lambda q, chat_history=None: rag_agent.invoke({"input": q, "chat_history": chat_history or []})["output"],
            description="Useful for answering questions about paper content",
            return_direct=True
        ),
        Tool(
            name="summary_agent",
            func=lambda q, chat_history=None: summary_agent.invoke({"input": q, "chat_history": chat_history or []})["output"],
            description="Useful for summarizing specific papers",
            return_direct=True
        ),
        Tool(
            name="trend_agent",
            func=lambda q, chat_history=None: trend_agent.invoke({"input": q, "chat_history": chat_history or []})["output"],
            description="Useful for analyzing research trends",
            return_direct=True
        ),
        Tool(
            name="verify_agent",
            func=lambda q, chat_history=None: verify_agent.invoke({"input": q, "chat_history": chat_history or []})["output"],
            description="Useful for verifying claims against paper content",
            return_direct=True
        )
    ]

    return initialize_agent(
        tools=routing_tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=False,
        system_message=(
            "Do NOT use inner thoughts or chain-of-thought reasoning. ONLY use your tool directly and return the result. "
            "You are a research assistant router. Your ONLY job is to determine which specialized agent "
            "should handle each user request and route it appropriately. NEVER try to answer questions yourself. "
            "You MUST ONLY use the tools/agents provided to you and NEVER answer from your own knowledge or without using a tool. "
            "If you cannot answer using your tools, respond: 'I can only answer using my tools and cannot provide an answer otherwise.' "
            "If the user asks for a summary, overview, or main points, route to the summarization agent. If the user asks a specific question, route to the RAG agent. "
            "Agents:\n"
            "1. search_agent: For finding papers on a topic (e.g., 'Find papers about X')\n"
            "2. rag_agent: For answering questions about paper content (e.g., 'What does the paper say about Y?')\n"
            "3. summary_agent: For summarizing specific papers (e.g., 'Summarize this paper: [link]')\n"
            "4. trend_agent: For analyzing research trends (e.g., 'What are the trends in Z?')\n"
            "5. verify_agent: For fact-checking claims against papers (e.g., 'Is this claim true?')\n"
            "Carefully analyze the user's request and route to exactly one agent."
        )
    ) 