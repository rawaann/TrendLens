import os
import requests
import time
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from typing import Dict, List, Any
from functools import lru_cache
from lxml import etree
import re
import PyPDF2
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------
# Load environment variables
# -----------------------

load_dotenv()

# -----------------------
# Initialize Ollama (try smaller model for speed)
# -----------------------

# Try llama3.2 first (faster), fallback to llama2 if not available
try:
    llm = OllamaLLM(model="llama3.2", temperature=0.3)
except:
    llm = OllamaLLM(model="llama2", temperature=0.3)

# -----------------------
# Cached ArXiv Search Function (Optimized)
# -----------------------

@lru_cache(maxsize=32)
def cached_arxiv_search(user_query, max_results=5):
    return arxiv_search(user_query, max_results)

def arxiv_search(user_query, max_results=5):
    terms = [f'all:"{word}"' for word in user_query.strip().split()]
    query_str = "+AND+".join(terms)

    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query={query_str}&start=0&max_results={max_results}"
    )
    print(f"🔍 Searching arXiv for: {user_query}")

    try:
        start_time = time.time()
        response = requests.get(url, timeout=3)  # Reduced timeout for speed
        duration = time.time() - start_time
        print(f"⏱️ ArXiv API call took {duration:.2f} seconds")

        response.raise_for_status()
        xml_data = response.text

        parser = etree.XMLParser(resolve_entities=False)
        root = etree.fromstring(xml_data.encode('utf-8'), parser=parser)

        namespaces = {
            "atom": "http://www.w3.org/2005/Atom",
            "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
            "arxiv": "http://arxiv.org/schemas/atom"
        }

        total_elem = root.find("opensearch:totalResults", namespaces)
        total_results = total_elem.text if total_elem is not None else "Unknown"
        print(f"📊 Total results found on arXiv: {total_results}")

        entries = root.findall("atom:entry", namespaces)

        if not entries:
            return []

        results = []
        for entry in entries:
            title_elem = entry.find("atom:title", namespaces)
            title_text = title_elem.text.strip() if (title_elem is not None and title_elem.text) else "No Title"

            summary_elem = entry.find("atom:summary", namespaces)
            summary_text = summary_elem.text.strip() if (summary_elem is not None and summary_elem.text) else "No Summary"

            link_elem = entry.find("atom:id", namespaces)
            link = link_elem.text if (link_elem is not None and link_elem.text) else "#"

            authors = []
            for author in entry.findall("atom:author", namespaces):
                name_elem = author.find("atom:name", namespaces)
                name = name_elem.text if (name_elem is not None and name_elem.text) else "Unknown Author"
                authors.append(name)
            authors_str = ", ".join(authors) if authors else "No Authors Listed"

            results.append({
                "title": title_text,
                "authors": authors_str,
                "abstract": summary_text[:300] + "..." if len(summary_text) > 300 else summary_text,
                "link": link
            })

        return results

    except requests.exceptions.RequestException as e:
        print(f"arXiv API request failed: {str(e)}")
        return []
    except etree.XMLSyntaxError as e:
        print(f"Failed to parse arXiv XML response: {str(e)}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during arXiv search: {str(e)}")
        return []

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
    try:
        response = requests.get(pdf_url, timeout=10)
        if response.status_code == 200:
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            return pdf_path
        else:
            print(f"Failed to download PDF for {arxiv_id}")
            return None
    except Exception as e:
        print(f"PDF download error for {arxiv_id}: {e}")
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

def process_paper_for_pdf(paper):
    link = paper.get('link')
    arxiv_id = extract_arxiv_id(link) if link else None
    if not arxiv_id:
        return {**paper, 'full_text': ''}
    pdf_path = download_pdf(arxiv_id)
    if not pdf_path:
        return {**paper, 'full_text': ''}
    text = extract_text_from_pdf(pdf_path)
    return {**paper, 'full_text': text}

# -----------------------
# Tool Configuration
# -----------------------

arxiv_tool = Tool(
    name="ArxivSearch",
    func=arxiv_search,
    description="Search arXiv.org for academic papers. Input should be a research query."
)

# -----------------------
# Agent Classes for Better Communication
# -----------------------

class PDFSummarizationAgent:
    """Agent for summarizing papers using full PDF text if available."""
    def __init__(self, llm):
        self.llm = llm

    def summarize_papers(self, papers: List[Dict], topic: str) -> List[Dict]:
        if not papers:
            return []
        # Download/extract PDFs in parallel
        print("\n📥 Downloading and extracting PDFs...")
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_paper_for_pdf, paper) for paper in papers]
            for future in as_completed(futures):
                results.append(future.result())
        # Summarize each paper
        print("\n📝 Summarizing each paper using PDF text (if available)...")
        summaries = []
        for paper in results:
            title = paper.get('title', 'No Title')
            link = paper.get('link', None)
            full_text = paper.get('full_text', '')
            if full_text:
                summary_input = f"Title: {title}\nLink: {link}\nFull Text: {full_text[:8000]}"  # Truncate for token limit
            else:
                summary_input = f"Title: {title}\nLink: {link}\n(No PDF text available. Summarize using title and abstract only.)\nAbstract: {paper.get('abstract', '')}"
            summarization_prompt = (
                f"Summarize the following paper in detail, including technical aspects, for a student audience.\n{summary_input}"
            )
            t0 = time.time()
            summary_result = self.llm.invoke(summarization_prompt)
            print(f"Summarization for '{title}' took {time.time() - t0:.2f} seconds")
            summaries.append({
                'title': title,
                'link': link,
                'summary': summary_result.strip(),
                'full_text': full_text
            })
        return summaries

class TrendDetectionAgent:
    """Agent for detecting trends from a set of papers."""
    def __init__(self, llm):
        self.llm = llm

    def detect_trends(self, papers: List[Dict], topic: str) -> str:
        if not papers:
            return "No papers available for trend detection."
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
        t0 = time.time()
        trend_result = self.llm.invoke(trend_prompt)
        print(f"Trend detection LLM call took {time.time() - t0:.2f} seconds")
        return trend_result.strip()

class ResearchCoordinator:
    """Coordinates the research workflow between different agents"""
    def __init__(self, llm):
        self.llm = llm
    def coordinate_research(self, topic: str) -> Dict[str, Any]:
        print(f"\n🎯 Research Coordinator: Starting research on '{topic}'")
        plan = self._create_research_plan(topic)
        print(f"📋 Research Plan: {plan}")
        results = self._execute_research_plan(topic, plan)
        return results
    def _create_research_plan(self, topic: str) -> str:
        prompt = f"""
        Create a brief research plan for studying '{topic}'. 
        Consider what specific aspects should be researched and how to structure the findings.
        Keep it concise and actionable.
        """
        t0 = time.time()
        response = self.llm.invoke(prompt)
        print(f"Research plan LLM call took {time.time() - t0:.2f} seconds")
        return response.strip()
    def _execute_research_plan(self, topic: str, plan: str) -> Dict[str, Any]:
        results = {
            "topic": topic,
            "plan": plan,
            "papers": None,
            "summaries": None,
            "trends": None,
            "fact_check": None,
            "timing": {}
        }
        start_time = time.time()
        # Phase 1: Paper Retrieval
        print("\n🔎 Phase 1: Paper Retrieval")
        retrieval_agent = PaperRetrievalAgent(self.llm)
        t0 = time.time()
        papers = retrieval_agent.retrieve_papers(topic)
        results["papers"] = papers
        results["timing"]["retrieval"] = time.time() - t0
        print("\n--- Retrieval Agent Output ---")
        for i, paper in enumerate(papers, 1):
            print(f"{i}. {paper['title']}\n   Authors: {paper['authors']}\n   Link: {paper['link']}\n")
        # Phase 2: PDF Summarization
        print("\n📝 Phase 2: PDF Summarization")
        pdf_summarization_agent = PDFSummarizationAgent(self.llm)
        t1 = time.time()
        summaries = pdf_summarization_agent.summarize_papers(papers, topic)
        results["summaries"] = summaries
        results["timing"]["pdf_summarization"] = time.time() - t1
        print("\n--- PDF Summarization Output ---")
        for summary in summaries:
            print(f"Summary for: {summary['title']}\n{summary['summary'][:500]}...\n")
        # Phase 2b: Trend Detection
        print("\n📈 Phase 2b: Trend Detection")
        trend_agent = TrendDetectionAgent(self.llm)
        t2 = time.time()
        trends = trend_agent.detect_trends(papers, topic)
        results["trends"] = trends
        results["timing"]["trend_detection"] = time.time() - t2
        print("\n--- Trend Detection Output ---")
        print(trends)
        # Phase 3: Fact Checking (on the concatenated summaries)
        print("\n🔍 Phase 3: Fact Checking")
        factcheck_agent = FactCheckAgent(self.llm)
        t3 = time.time()
        all_summaries_text = "\n\n".join([s['summary'] for s in summaries])
        fact_check = factcheck_agent.fact_check_summary(all_summaries_text, papers)
        results["fact_check"] = fact_check
        results["timing"]["fact_check"] = time.time() - t3
        print("\n--- Fact-Check Agent Output ---")
        print(fact_check)
        results["timing"]["total"] = time.time() - start_time
        return results

class PaperRetrievalAgent:
    """Specialized agent for retrieving academic papers"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def retrieve_papers(self, topic: str) -> List[Dict]:
        """Retrieves relevant papers for the given topic"""
        print(f"📚 Retrieving papers for: {topic}")
        
        # Use cached arXiv search for speed
        papers = cached_arxiv_search(topic, 5)
        
        if not papers:
            print("❌ No papers retrieved")
            return []
        
        print(f"✅ Retrieved {len(papers)} papers")
        return papers

class FactCheckAgent:
    """Specialized agent for fact-checking summaries"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def fact_check_summary(self, summary: str, papers: List[Dict]) -> str:
        """Fact-checks the summary against the original papers"""
        if not papers:
            return "No papers available for fact-checking."
        
        # Format papers for comparison
        papers_text = self._format_papers_for_factcheck(papers)
        
        prompt = f"""
        You are an academic fact-checker. Review this summary against the original papers and provide corrections.
        
        Summary to fact-check:
        {summary}
        
        Original papers:
        {papers_text}
        
        Instructions:
        - Identify any factual inaccuracies or misrepresentations
        - Correct any errors you find
        - Ensure all claims are supported by the papers
        - Cite specific papers when making corrections
        - If the summary is accurate, confirm this
        - Provide a corrected version if needed
        """
        
        t0 = time.time()
        fact_check = self.llm.invoke(prompt)
        print(f"Fact-check LLM call took {time.time() - t0:.2f} seconds")
        return fact_check.strip()
    
    def _format_papers_for_factcheck(self, papers: List[Dict]) -> str:
        """Formats papers for fact-checking comparison"""
        formatted = []
        for i, paper in enumerate(papers, 1):
            formatted.append(f"""
            Paper {i}: {paper['title']}
            Authors: {paper['authors']}
            Abstract: {paper['abstract']}
            Link: {paper['link']}
            """)
        return "\n".join(formatted)

# -----------------------
# Main Research Workflow
# -----------------------

if __name__ == "__main__":
    topic = "physics learning education"
    
    try:
        # Initialize the research coordinator
        coordinator = ResearchCoordinator(llm)
        
        # Execute the complete research workflow
        results = coordinator.coordinate_research(topic)
        
        # Display results
        print("\n" + "="*60)
        print("🎯 RESEARCH RESULTS")
        print("="*60)
        
        print(f"\n📋 Research Plan:")
        print(results["plan"])
        
        print(f"\n📚 Retrieved Papers ({len(results['papers'])}):")
        for i, paper in enumerate(results["papers"], 1):
            print(f"  {i}. {paper['title']}")
            print(f"     Authors: {paper['authors']}")
            print(f"     Link: {paper['link']}")
            print()
        
        print(f"\n📝 Summaries:")
        for i, summary in enumerate(results["summaries"], 1):
            print(f"  {i}. Summary for: {summary['title']}")
            print(f"     Link: {summary['link']}")
            print(f"     Summary: {summary['summary'][:200]}...")
            print()

        print(f"\n📈 Trends:")
        print(results["trends"])
        
        print(f"\n🔍 Fact-Check Results:")
        print(results["fact_check"])
        
        print(f"\n⏱️ Performance Metrics:")
        for phase, duration in results["timing"].items():
            print(f"  {phase.capitalize()}: {duration:.2f} seconds")
        
        print(f"\n✅ Total Research Time: {results['timing']['total']:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error during research: {str(e)}")
        import traceback
        traceback.print_exc()
