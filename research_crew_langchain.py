import os
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory

# -----------------------
# Load environment variables
# -----------------------

load_dotenv()

# -----------------------
# Initialize Ollama
# -----------------------

llm = OllamaLLM(model="llama2", temperature=0.3)

# -----------------------
# ArXiv Search Function
# -----------------------

def arxiv_search(user_query, max_results=5):
    terms = [f'all:"{word}"' for word in user_query.strip().split()]
    query_str = "+AND+".join(terms)

    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query={query_str}&start=0&max_results={max_results}"
    )
    print(f"Requesting URL:\n{url}\n")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        xml_data = response.text

        root = ET.fromstring(xml_data)

        namespaces = {
            "atom": "http://www.w3.org/2005/Atom",
            "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
            "arxiv": "http://arxiv.org/schemas/atom"
        }

        total_elem = root.find("opensearch:totalResults", namespaces)
        total_results = total_elem.text if total_elem is not None else "Unknown"
        print(f"🔎 Total results found on arXiv: {total_results}")

        entries = root.findall("atom:entry", namespaces)

        if not entries:
            return "No arXiv papers found."

        results = []
        for entry in entries:
            title_elem = entry.find("atom:title", namespaces)
            title_text = title_elem.text.strip() if title_elem is not None else "No Title"

            summary_elem = entry.find("atom:summary", namespaces)
            summary_text = summary_elem.text.strip() if summary_elem is not None else "No Summary"

            link_elem = entry.find("atom:id", namespaces)
            link = link_elem.text if link_elem is not None else "#"

            authors = []
            for author in entry.findall("atom:author", namespaces):
                name_elem = author.find("atom:name", namespaces)
                name = name_elem.text if name_elem is not None else "Unknown Author"
                authors.append(name)
            authors_str = ", ".join(authors) if authors else "No Authors Listed"

            results.append(
                f"• {title_text}\n"
                f"  Authors: {authors_str}\n"
                f"  Abstract: {summary_text[:300]}...\n"
                f"  Link: {link}\n"
            )

        return "\n".join(results)

    except requests.exceptions.RequestException as e:
        return f"arXiv API request failed: {str(e)}"
    except ET.ParseError as e:
        return f"Failed to parse arXiv XML response: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred during arXiv search: {str(e)}"

# -----------------------
# Tool Configuration
# -----------------------

tools = [
    Tool(
        name="ArxivSearch",
        func=arxiv_search,
        description="Search arXiv.org for academic papers. Input should be a research query."
    )
]

# -----------------------
# Memory Setup
# -----------------------

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output"
)

# -----------------------
# Agent Configuration
# -----------------------

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
    max_iterations=3,
    agent_kwargs={
        "prefix": """
If asked for academic papers, use the ArxivSearch tool with the search query.
Never make up references.
"""
    }
)

# -----------------------
# Full Research Workflow
# -----------------------

if __name__ == "__main__":
    topic = "physics learning education"

    try:
        # ✅ PHASE 1 - RUN TOOL DIRECTLY
        print("\n🔎 Phase 1: Searching arXiv...\n")
        research = arxiv_search(topic, max_results=5)
        print("\n📚 Research Results:\n", research)

        # ✅ PHASE 2 - Verification via LLM
        print("\n🔍 Phase 2: Verifying...\n")
        verification_prompt = (
            f"Check if these papers about {topic} are consistent and highlight discrepancies:\n\n{research}"
        )
        verification = llm.invoke(verification_prompt)
        print("\n✅ Verification Results:\n", verification)

        # ✅ PHASE 3 - Summarizing
        print("\n🧠 Phase 3: Summarizing...\n")
        summary_prompt = (
            f"Summarize these findings into a 3-5 paragraph article for students:\n\n{research}"
        )
        summary = llm.invoke(summary_prompt)
        print("\n📄 Summary:\n", summary)

        # ✅ PHASE 4 - Fact-checking
        print("\n🛡️ Phase 4: Fact-checking the summary...\n")
        fact_check_prompt = (
            f"Fact-check this summary. Correct inaccuracies and cite arXiv sources if possible:\n\n{summary}"
        )
        fact_check = llm.invoke(fact_check_prompt)
        print("\n✅ Final Fact-Checked Summary:\n", fact_check)

    except Exception as e:
        print(f"Error: {str(e)}")
