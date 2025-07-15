from langchain.agents import initialize_agent, AgentType, Tool
from tools.summarization_tools import summarize_paper_tool
from config import llm

def create_summarization_agent():
    summary_tools = [
        Tool(
            name="summarize_paper",
            func=summarize_paper_tool,
            description="ONLY provide comprehensive summaries or overviews of entire papers. NOT for answering specific questions. Example input: 'Summarize the first paper.' Example output: [summary from summarize_paper tool] Do NOT use or invent any other tool names.",
            return_direct=True
        )
    ]
    return initialize_agent(
        tools=summary_tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=False,
        system_message=(
            "Do NOT use inner thoughts or chain-of-thought reasoning. ONLY use your tool directly and return the result. "
            "NEVER use the RAG tool for summarization. ONLY use the summarize_paper tool for summaries, overviews, or main points. "
            "You are a paper summarization specialist. Your ONLY job is to provide comprehensive summaries or overviews of entire academic papers. "
            "If the user asks for a summary, overview, or main points of a paper, use ONLY the 'summarize_paper' tool. "
            "NEVER use this tool for answering specific questions. ONLY use for full summaries or overviews. Do NOT use or invent any other tool names. "
            "If the user asks for something outside this tool, respond: 'I can only summarize papers using the provided tool.' "
            "You MUST ONLY use the tools provided to you and NEVER answer from your own knowledge or without using a tool. "
            "If you cannot answer using your tools, respond: 'I can only answer using my tools and cannot provide an answer otherwise.' "
            "Example input: 'Summarize the first paper.'\n"
            "Example output: [summary from summarize_paper tool]"
        )
    ) 