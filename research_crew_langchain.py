import os
import requests
import xml.etree.ElementTree as ET
import time
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

ollama_llm = OllamaLLM(model="llama2", temperature=0.3)

# -----------------------
# ArXiv Search Tool
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
        start_time = time.time()
        response = requests.get(url, timeout=10)
        duration = time.time() - start_time
        print(f"⏱️ ArxivSearch API call took {duration:.2f} seconds")

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
            return "Final Answer: No arXiv papers found."

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

        result_text = "\n".join(results)
        return result_text

    except requests.exceptions.RequestException as e:
        return f"arXiv API request failed: {str(e)}"
    except ET.ParseError as e:
        return f"Failed to parse arXiv XML response: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred during arXiv search: {str(e)}"

# -----------------------
# Tool Configuration
# -----------------------

arxiv_tool = Tool(
    name="ArxivSearch",
    func=arxiv_search,
    description="Search arXiv.org for academic papers. Input should be a research query."
)

# -----------------------
# Retrieval Agent
# -----------------------

retrieval_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output"
)

retrieval_agent = initialize_agent(
    tools=[arxiv_tool],
    llm=ollama_llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=retrieval_memory,
    handle_parsing_errors=True,
    max_iterations=1,
    agent_kwargs={
        "prefix": """
You are an academic research assistant whose only job is to run the ArxivSearch tool
once to retrieve up to 5 academic papers matching the user's topic.
- Do not summarize.
- Do not analyze.
- Do not invent any papers.
"""
    }
)

# -----------------------
# Summarization Agent
# -----------------------

summarization_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output"
)

summarization_agent = initialize_agent(
    tools=[],
    llm=ollama_llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=summarization_memory,
    handle_parsing_errors=True,
    max_iterations=1,
    agent_kwargs={
        "prefix": """
You are an expert science writer. Your only task is:
- Summarize the provided list of academic papers into a 3-5 paragraph article for students.
- Do not invent new papers.
- Do not cite extra references.
- Do not perform fact-checking.
"""
    }
)

# -----------------------
# Fact-Checking Agent
# -----------------------

factcheck_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output"
)

factcheck_agent = initialize_agent(
    tools=[],
    llm=ollama_llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=factcheck_memory,
    handle_parsing_errors=True,
    max_iterations=1,
    agent_kwargs={
        "prefix": """
You are an academic fact-checker.
- Your only job is to fact-check the provided summary.
- Correct any inaccuracies you find.
- Cite arXiv links where possible.
- Do not summarize papers again.
"""
    }
)

# -----------------------
# Full Pipeline Workflow
# -----------------------

if __name__ == "__main__":
    topic = "physics learning education"

    try:
        # STEP 1 — Retrieval
        print("\n🔎 Phase 1 — Retrieval...\n")

        retrieval_prompt = (
            f"Find up to 5 academic papers about {topic}. "
            f"Use ArxivSearch only once. "
            f"Return the list of papers as Final Answer."
        )

        retrieval_result = retrieval_agent.invoke({"input": retrieval_prompt}).get("output")
        print("\n✅ Retrieved Papers:\n", retrieval_result)

        # STEP 2 — Summarization
        print("\n📝 Phase 2 — Summarization...\n")

        summarization_prompt = (
            f"Summarize these papers into a 3-5 paragraph article for students:\n\n{retrieval_result}"
        )

        summary_result = summarization_agent.invoke({"input": summarization_prompt}).get("output")
        print("\n✅ Summary:\n", summary_result)

        # STEP 3 — Fact-Checking
        print("\n🔍 Phase 3 — Fact-Checking...\n")

        factcheck_prompt = (
            f"Fact-check this summary. Correct any inaccuracies and cite arXiv links where possible:\n\n{summary_result}"
        )

        factcheck_result = factcheck_agent.invoke({"input": factcheck_prompt}).get("output")
        print("\n✅ Fact-Checked Summary:\n", factcheck_result)

    except Exception as e:
        print(f"Error: {str(e)}")
