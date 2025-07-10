import os
import re
import requests
import xml.etree.ElementTree as ET
import PyPDF2
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
# PDF Download/Extraction Helpers
# -----------------------

CACHE_DIR = "pdf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def extract_arxiv_id(link):
    match = re.search(r'arxiv.org/(abs|pdf)/([\w.]+)', link)
    return match.group(2) if match else None

def download_pdf(arxiv_id):
    pdf_path = os.path.join(CACHE_DIR, f"{arxiv_id}.pdf")
    if os.path.exists(pdf_path):
        # Already cached
        return pdf_path
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        return pdf_path
    else:
        print(f"Failed to download PDF for {arxiv_id}")
        return None

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"PDF extraction error for {pdf_path}: {e}")
        return ""

def process_paper(paper):
    link = paper.get('link')
    arxiv_id = extract_arxiv_id(link) if link else None
    if not arxiv_id:
        return paper['title'], link, ""
    pdf_path = download_pdf(arxiv_id)
    if not pdf_path:
        return paper['title'], link, ""
    text = extract_text_from_pdf(pdf_path)
    return paper['title'], link, text

# -----------------------
# Full Research Workflow
# -----------------------

def run_research(topic):
    start_time = time.time()
    output = []
    try:
        # ✅ PHASE 1 - RUN TOOL DIRECTLY
        output.append("\n🔎 Phase 1: Searching arXiv...\n")
        retrieval_result = arxiv_search(topic, max_results=5)
        output.append("\n📚 Research Results:\n" + retrieval_result)

        # STEP 2 — Per-Paper Summarization (Full PDF)
        output.append("\n📝 Phase 2 — Per-Paper Summarization (Full PDF)...\n")

        # Parse papers from retrieval_result
        papers = []
        current = {}
        for line in retrieval_result.splitlines():
            if line.startswith('• '):
                if current:
                    papers.append(current)
                current = {'title': line[2:].strip()}
            elif line.strip().startswith('Abstract:'):
                current['abstract'] = line.strip().split('Abstract:')[1].strip().rstrip('.')
            elif line.strip().startswith('Link:'):
                current['link'] = line.strip().split('Link:')[1].strip()
        if current:
            papers.append(current)

        # Parallel PDF download/extraction
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_paper, paper) for paper in papers]
            for future in as_completed(futures):
                title, link, full_text = future.result()
                results.append({'title': title, 'link': link, 'full_text': full_text})

        # Summarize each paper using the full text
        for paper in results:
            title = paper.get('title', 'No Title')
            link = paper.get('link', None)
            full_text = paper.get('full_text', '')
            if not full_text:
                summary_input = f"Title: {title}\nLink: {link}\n(No PDF text available. Summarize using title only.)"
            else:
                summary_input = f"Title: {title}\nLink: {link}\nFull Text: {full_text[:8000]}"  # Truncate to avoid token limits
            summarization_prompt = (
                f"Summarize the following paper in detail, including technical aspects, for a student audience.\n{summary_input}"
            )
            summary_result = llm.invoke(summarization_prompt)
            output.append(f"\n---\nSummary for: {title}\nPDF: {link}\n\n{summary_result}\n---\n")

        # ✅ PHASE 3 - Fact-checking
        output.append("\n🛡️ Phase 3: Fact-checking the summary...\n")
        fact_check_prompt = (
            f"Fact-check this summary. Correct inaccuracies and cite arXiv sources if possible:\n\n{summary_result}"
        )
        fact_check = llm.invoke(fact_check_prompt)
        output.append("\n✅ Final Fact-Checked Summary:\n" + str(fact_check))

        # Trend Detection Step
        output.append("\n📈 Phase 2b — Trend Detection in Topic...\n")
        trend_input = ""
        for paper in papers:
            title = paper.get('title', 'No Title')
            abstract = paper.get('abstract', '')
            trend_input += f"Title: {title}\nAbstract: {abstract}\n\n"
        trend_prompt = (
            f"Analyze the following list of academic papers about {topic}. "
            "Identify the most common research topics, emerging trends, and any notable shifts in focus. "
            "Summarize your findings as a list of trends with brief explanations.\n\n"
            f"{trend_input}"
        )
        trend_result = llm.invoke(trend_prompt)
        output.append("\n📈 Trend Detection Result:\n" + str(trend_result))
        end_time = time.time()  # End timer
        elapsed = end_time - start_time
        output.append(f"\n⏱️ Total runtime: {elapsed:.2f} seconds")

    except Exception as e:
        output.append(f"Error: {str(e)}")
    return "\n".join(output)