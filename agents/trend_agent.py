from langchain.agents import initialize_agent, AgentType, Tool
from tools.trend_tools import detect_trends_tool, analyze_trends_in_topic
from config import llm

def create_trend_agent():
    trend_tools = [
        Tool(
            name="detect_trends",
            func=detect_trends_tool,
            description="Analyze research trends in a set of papers using only this tool. Input: text or abstracts of papers. Output: structured trend analysis. Do NOT use or invent any other tool names.",
            return_direct=True
        ),
        Tool(
            name="analyze_trends_in_topic",
            func=analyze_trends_in_topic,
            description="Search for and analyze trends in a research topic using only this tool. Input: topic string. Output: structured trend analysis. Do NOT use or invent any other tool names.",
            return_direct=True
        )
    ]
    return initialize_agent(
        tools=trend_tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=False,
        system_message=(
            "Do NOT use inner thoughts or chain-of-thought reasoning. ONLY use your tool directly and return the result. "
            "You are a research trend analyst. You can ONLY use the 'detect_trends' and 'analyze_trends_in_topic' tools. "
            "If the user asks for trend analysis, use ONLY these tools. Do NOT use or invent any other tool names. "
            "If the user asks for something outside these tools, respond: 'I can only analyze trends using the provided tools.' "
            "You MUST ONLY use the tools provided to you and NEVER answer from your own knowledge or without using a tool. "
            "If you cannot answer using your tools, respond: 'I can only answer using my tools and cannot provide an answer otherwise.' "
            "Example input: 'What are the trends in quantum computing?'\n"
            "Example output: [structured trend analysis from detect_trends or analyze_trends_in_topic]"
        )
    ) 