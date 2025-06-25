import os
import requests
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Initialize Ollama
llm = OllamaLLM(model="llama2", temperature=0.3)  

# Enhanced Serper search function
def serper_search(query):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    data = {"q": query, "num": 5}
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        results = response.json()
        if not results.get("organic"):
            return "No results found"
        output = []
        for item in results["organic"][:5]:
            title = item.get('title', 'No title')
            snippet = item.get('snippet', 'No description')
            link = item.get('link', '#')
            output.append(f"• {title}\n  {snippet}\n  Source: {link}\n")
        return "\n".join(output)
    except Exception as e:
        return f"Search failed: {str(e)}"

# Tool configuration
tools = [
    Tool(
        name="SerperSearch",
        func=serper_search,
        description="Useful for searching the web about current topics. Input should be a clear search query."
    )
]

# Memory setup
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output"
)

# Agent with strict formatting
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
    max_iterations=3,
    agent_kwargs={
        "prefix": """You are a research assistant that MUST use the SerperSearch tool when asked about current information.
        Follow these rules STRICTLY:
        1. ALWAYS use this format for tools:
        Action:
        ```
        {{
            "action": "SerperSearch",
            "action_input": "your search query here"
        }}
        ```
        2. Never make up answers - always use the tool
        3. Keep queries concise but specific"""
    }
)

# Full workflow
if __name__ == "__main__":
    topic = "recent developments in AI transforming education 2024"

    try:
        print("🔎 Phase 1: Researching...")
        research = agent.run(
            f"Use SerperSearch to find 5 concrete examples of {topic}. "
            "For each result, include: (1) The development (2) How it's being used (3) Source URL"
        )
        print("\n📝 Research Results:\n", research)

        print("\n🔍 Phase 2: Verifying...")
        verification = agent.run(
            f"Verify these claims about {topic} by searching for supporting evidence:\n{research}"
        )
        print("\n✅ Verification Results:\n", verification)

        print("\n🧠 Phase 3: Summarizing...")
        summary_prompt = f"Summarize the following research into a clear 3–5 paragraph article for students:\n{research}"
        summary = llm.invoke(summary_prompt)
        print("\n📄 Summary:\n", summary)

        print("\n🛡️ Phase 4: Fact-checking the summary...")
        fact_check_prompt = f"Fact-check this summary. Correct any inaccuracies and cite supporting sources if possible:\n{summary}"
        fact_check = llm.invoke(fact_check_prompt)
        print("\n✅ Final Fact-Checked Summary:\n", fact_check)

    except Exception as e:
        print(f"Error: {str(e)}")
