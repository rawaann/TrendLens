import os
import re
import requests
import time
from dotenv import load_dotenv
from functools import lru_cache
import PyPDF2
from typing import Optional

# LangChain imports
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# -----------------------
# Configuration
# -----------------------

load_dotenv()

HUGGINGFACE_TOKEN = "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"
OLLAMA_MODEL = "llama3.2"
ARXIV_MAX_RESULTS = 5
PDF_CHUNK_SIZE = 500
PDF_CHUNK_OVERLAP = 100
CACHE_DIR = "pdf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# -----------------------
# Initialize Models
# -----------------------

try:
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.3)
except:
    llm = OllamaLLM(model="llama2", temperature=0.3)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -----------------------
# Core Functions
# -----------------------

@lru_cache(maxsize=32)
def cached_arxiv_search(user_query, max_results=ARXIV_MAX_RESULTS):
    return arxiv_search(user_query, max_results)

def arxiv_search(user_query, max_results=ARXIV_MAX_RESULTS):
    from lxml import etree  # Import etree at the top of the function for linter
    terms = [f'all:"{word}"' for word in user_query.strip().split()]
    query_str = "+AND+".join(terms)
    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query={query_str}&start=0&max_results={max_results}"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        xml_data = response.text
        parser = etree.XMLParser(resolve_entities=False)
        root = etree.fromstring(xml_data.encode('utf-8'), parser=parser)
        namespaces = {
            "atom": "http://www.w3.org/2005/Atom",
            "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
            "arxiv": "http://arxiv.org/schemas/atom"
        }
        entries = root.findall("atom:entry", namespaces)
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
    except Exception:
        return []

def extract_arxiv_id(link):
    match = re.search(r'arxiv.org/(abs|pdf)/([\w.]+)', link)
    return match.group(2) if match else None

def download_pdf(arxiv_id):
    pdf_path = os.path.join(CACHE_DIR, f"{arxiv_id}.pdf")
    if os.path.exists(pdf_path):
        return pdf_path
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        response = requests.get(pdf_url, timeout=10)
        if response.status_code == 200:
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            return pdf_path
        else:
            return None
    except Exception:
        return None

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception:
        return ""

def process_paper_for_pdf(paper):
    link = paper.get('link')
    arxiv_id = extract_arxiv_id(link) if link else None
    pdf_path = download_pdf(arxiv_id) if arxiv_id else None
    if pdf_path:
        paper['pdf_path'] = pdf_path
        text = extract_text_from_pdf(pdf_path)
        paper['full_text'] = text
    else:
        paper['pdf_path'] = None
        paper['full_text'] = ''
    return paper

def summarize_text_with_longt5(text, title="", link=""):
    try:
        if len(text.strip()) < 100:
            return f"Text too short to summarize: {text[:200]}..."
        prompt = f"""
        Please provide a comprehensive summary of the following academic paper.
        Title: {title}
        Link: {link}
        Paper Text:
        {text[:8000]}
        Please provide a detailed summary that includes:
        1. Main research objectives
        2. Key findings and results
        3. Methodology used
        4. Conclusions and implications
        5. Technical details relevant to the field
        Summary:
        """
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        print(f"Summarization error: {e}")
        return f"Error summarizing text: {str(e)}"

def build_vectorstore_from_papers(papers, embedding_model=None):
    print(f"[VSTORE] Building vectorstore from {len(papers)} papers...")
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PDF_CHUNK_SIZE, 
        chunk_overlap=PDF_CHUNK_OVERLAP
    )
    from langchain.schema import Document
    all_documents = []
    chunk_to_paper = []
    for idx, paper in enumerate(papers):
        full_text = paper.get('full_text', '')
        if not full_text:
            print(f"[VSTORE] Paper {idx+1} has no full text, skipping.")
            continue
        chunks = text_splitter.split_text(full_text)
        print(f"[VSTORE] Paper {idx+1}: '{paper.get('title','Unknown')}' split into {len(chunks)} chunks.")
        for chunk_idx, chunk in enumerate(chunks):
            all_documents.append(Document(
                page_content=chunk,
                metadata={
                    "paper_title": paper.get('title', 'Unknown Title'),
                    "arxiv_id": extract_arxiv_id(paper.get('link', '')),
                    "chunk_idx": chunk_idx,
                    "paper_idx": idx
                }
            ))
            chunk_to_paper.append(paper)
    print(f"[VSTORE] Total chunks to embed: {len(all_documents)}")
    if not all_documents:
        print("[VSTORE] No chunks to embed. Returning None.")
        return None, None
    print("[VSTORE] Embedding all chunks...")
    vectorstore = FAISS.from_documents(all_documents, embedding_model)
    print("[VSTORE] Vectorstore built successfully.")
    return vectorstore, chunk_to_paper

def rag_qa(vectorstore, question, k=5):
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        rag_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are an academic assistant. Given the following context chunks from research papers, "
                "do your best to answer the user's question as accurately as possible. If the answer is not fully contained, "
                "provide an approximate answer based on the information available in these chunks. "
                "Context Chunks:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            )
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": rag_prompt
            }
        )
        result = qa_chain.invoke({"query": question})
        return result["result"], result.get("source_documents", [])
    except Exception as e:
        return f"Error answering question: {str(e)}", []

# Global state to store last searched papers for RAG
last_searched_papers = []
# Add global variables for vectorstore and chunk mapping
last_vectorstore = None
last_chunk_to_paper = None

def search_papers_tool(topic: str) -> str:
    global last_searched_papers, last_vectorstore, last_chunk_to_paper
    print(f"[TOOL] search_papers_tool called with topic: {topic}")
    papers = arxiv_search(topic, ARXIV_MAX_RESULTS)
    if papers is None:
        papers = []
    if not papers:
        print("[TOOL] No papers found.")
        return f"No papers found for topic: {topic}"
    print(f"[TOOL] Downloading and processing {len(papers)} papers...")
    processed_papers = []
    for idx, paper in enumerate(papers):
        print(f"[TOOL] Processing paper {idx+1}: {paper['title']}")
        arxiv_id = extract_arxiv_id(paper.get('link', ''))
        pdf_path = None
        if arxiv_id:
            pdf_path = os.path.join(CACHE_DIR, f"{arxiv_id}.pdf")
            if not os.path.exists(pdf_path):
                _ = download_pdf(arxiv_id)
        if pdf_path and os.path.exists(pdf_path):
            paper['pdf_path'] = pdf_path
        else:
            paper['pdf_path'] = None
        processed_papers.append(process_paper_for_pdf(paper))
    last_searched_papers = processed_papers
    # Embed and cache vectorstore and chunk mapping
    last_vectorstore, last_chunk_to_paper = build_vectorstore_from_papers(processed_papers)
    # Format output as a clean, readable list
    result = f"Found {len(processed_papers)} papers for '{topic}':\n\n"
    for i, paper in enumerate(processed_papers, 1):
        result += f"{i}. {paper.get('title', 'Unknown Title')}\n"
        result += f"   Authors: {paper.get('authors', 'Unknown Authors')}\n"
        result += f"   Link: {paper.get('link', 'No Link')}\n"
        if paper.get('pdf_path'):
            result += f"   PDF: {paper['pdf_path']}\n"
        abstract = paper.get('abstract', '')
        if abstract:
            result += f"   Abstract: {abstract[:400]}{'...' if len(abstract) > 400 else ''}\n"
        result += "\n"
    return result

def get_last_searched_papers_context():
    """
    Returns a formatted string with the last searched topic, paper titles, authors, abstracts, and PDF file paths (if available).
    """
    global last_searched_papers
    if not last_searched_papers:
        return "No papers have been searched yet."
    context = "Last searched papers:\n"
    for i, paper in enumerate(last_searched_papers, 1):
        context += f"{i}. {paper.get('title', 'Unknown Title')}\n"
        context += f"   Authors: {paper.get('authors', 'Unknown Authors')}\n"
        context += f"   Abstract: {paper.get('abstract', '')[:200]}{'...' if len(paper.get('abstract', '')) > 200 else ''}\n"
        context += f"   Link: {paper.get('link', 'No Link')}\n"
        pdf_path = None
        if 'link' in paper and paper['link']:
            arxiv_id = extract_arxiv_id(paper['link'])
            if arxiv_id:
                pdf_path = os.path.join(CACHE_DIR, f"{arxiv_id}.pdf")
        if pdf_path and os.path.exists(pdf_path):
            context += f"   PDF: {pdf_path}\n"
        context += "\n"
    return context

# Update RAG, summarization, and trend tools to use this context in their prompts

def rag_tool(question: str) -> str:
    print(f"[TOOL] rag_tool called with question: {question}")
    global last_searched_papers, last_vectorstore, last_chunk_to_paper
    if not last_searched_papers:
        print("[TOOL] No papers available for RAG.")
        return "No papers available. Please search for papers first using the 'search' command."
    processed_papers = last_searched_papers
    if processed_papers is None:
        processed_papers = []
    # Use cached vectorstore if available
    vectorstore = last_vectorstore
    chunk_to_paper = last_chunk_to_paper
    if not vectorstore or not chunk_to_paper:
        print("[TOOL] No cached vectorstore, building new one...")
        vectorstore, chunk_to_paper = build_vectorstore_from_papers(processed_papers)
    if chunk_to_paper is None:
        chunk_to_paper = []
    if not vectorstore:
        print("[TOOL] No full text available for RAG.")
        return "No full text available for RAG. Try searching for different papers."
    print(f"[TOOL] Running RAG QA with question: {question}")
    answer, sources = rag_qa(vectorstore, question)
    retrieved_context = ""
    for i, doc in enumerate(sources, 1):
        chunk_text = doc.page_content if hasattr(doc, "page_content") else str(doc)
        meta = getattr(doc, 'metadata', {})
        title = meta.get('paper_title', 'Unknown Title')
        arxiv_id = meta.get('arxiv_id', 'Unknown')
        chunk_idx = meta.get('chunk_idx', None)
        retrieved_context += f"Source {i} (Title: {title}, arXiv ID: {arxiv_id}, Chunk: {chunk_idx}):\n{chunk_text[:500]}{'...' if len(chunk_text) > 500 else ''}\n\n"
    last_searched_papers = processed_papers
    return f"Retrieved Context (from PDF chunks):\n{retrieved_context}\n\nAnswer: {answer}"


def summarize_paper_tool(paper_url: str) -> str:
    print(f"[TOOL] summarize_paper_tool called with URL: {paper_url}")
    global last_searched_papers
    try:
        arxiv_id = extract_arxiv_id(paper_url)
        if not arxiv_id:
            print("[TOOL] Invalid arXiv URL.")
            return "Invalid arXiv URL. Please provide a valid arXiv link."
        paper = {"link": paper_url, "title": f"Paper {arxiv_id}"}
        print(f"[TOOL] Downloading and processing paper: {arxiv_id}")
        processed_paper = process_paper_for_pdf(paper)
        if processed_paper is None:
            processed_paper = {}
        if not processed_paper.get('full_text'):
            print("[TOOL] Could not download or extract text from PDF.")
            return "Could not download or extract text from the PDF."
        print(f"[TOOL] Summarizing paper: {processed_paper.get('title', 'Unknown Title')}")
        context = get_last_searched_papers_context()
        summary = summarize_text_with_longt5(
            processed_paper['full_text'], 
            processed_paper.get('title', 'Unknown Title'),
            paper_url
        )
        last_searched_papers = [processed_paper]
        return f"Context:\n{context}\n\nSummary of {processed_paper.get('title', 'Unknown Title')}:\n\n{summary}"
    except Exception as e:
        print(f"[TOOL] Error summarizing paper: {e}")
        return f"Error summarizing paper: {str(e)}"


def detect_trends_tool(papers_text: str) -> str:
    print(f"[TOOL] detect_trends_tool called.")
    try:
        prompt = f"""
        Analyze the following academic papers and identify the main research trends, 
        common themes, and emerging topics. Provide a structured analysis with key findings.
        Papers:
        {papers_text}
        Please provide:
        1. Main research themes
        2. Emerging trends
        3. Key findings
        4. Future research directions
        """
        print("[TOOL] Invoking LLM for trend detection...")
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        print(f"[TOOL] Error detecting trends: {e}")
        return f"Error detecting trends: {str(e)}"


def summarize_multiple_papers_tool(papers_text: str) -> str:
    global last_searched_papers
    try:
        context = get_last_searched_papers_context()
        if not papers_text and last_searched_papers:
            papers_text = "\n\n".join([p.get('full_text', '') for p in last_searched_papers if p.get('full_text')])
        return f"Context:\n{context}\n\nTo summarize papers, please use the 'summarize_paper' tool with specific arXiv URLs.\n        \nExample: summarize_paper https://arxiv.org/abs/2208.00733v1\n        \nOr use the 'detect_trends' tool to analyze the papers you found."
    except Exception as e:
        return f"Error summarizing papers: {str(e)}"


def analyze_trends_in_topic(topic: str) -> str:
    print(f"[TOOL] analyze_trends_in_topic called with topic: {topic}")
    global last_searched_papers
    try:
        papers = arxiv_search(topic, ARXIV_MAX_RESULTS)
        if papers is None:
            papers = []
        if not papers:
            print("[TOOL] No papers found for trend analysis.")
            return f"No papers found for topic: {topic}"
        if papers is None:
            papers = []
        context = get_last_searched_papers_context()
        papers_text = f"Papers on '{topic}':\n\n"
        for i, paper in enumerate(papers, 1):
            papers_text += f"Paper {i}: {paper['title']}\n"
            papers_text += f"Authors: {paper['authors']}\n"
            papers_text += f"Abstract: {paper['abstract']}\n\n"
        print("[TOOL] Calling detect_trends_tool...")
        trends = detect_trends_tool(papers_text)
        last_searched_papers = papers
        return f"Context:\n{context}\n\nTrend Analysis for '{topic}':\n\n{trends}"
    except Exception as e:
        print(f"[TOOL] Error analyzing trends: {e}")
        return f"Error analyzing trends for topic '{topic}': {str(e)}"

def verify_information_tool(claim: str, paper_url: Optional[str] = None) -> str:
    """
    Verifies the given claim/summary/answer against the full text of the specified paper.
    If paper_url is None, uses the last searched papers.
    """
    print(f"[TOOL] verify_information_tool called with claim: {claim} and paper_url: {paper_url}")
    try:
        papers_to_check = []
        if paper_url:
            arxiv_id = extract_arxiv_id(paper_url)
            if not arxiv_id:
                return "Invalid arXiv URL for verification."
            paper = {"link": paper_url, "title": f"Paper {arxiv_id}"}
            processed_paper = process_paper_for_pdf(paper)
            if not processed_paper.get('full_text'):
                return "Could not download or extract text from the PDF for verification."
            papers_to_check = [processed_paper]
        else:
            global last_searched_papers
            if not last_searched_papers:
                return "No papers available for verification. Please search for papers first."
            papers_to_check = last_searched_papers

        # Concatenate all full texts for verification
        all_text = "\n\n".join([p.get('full_text', '') for p in papers_to_check if p.get('full_text')])
        if not all_text.strip():
            return "No full text available for verification."

        prompt = f"""
        Please verify the following claim or summary against the provided academic paper text.
        Claim/Summary:
        {claim}

        Paper Text:
        {all_text[:8000]}

        Indicate whether the claim is supported, contradicted, or unverifiable based on the paper. Provide evidence or quotes from the paper to support your assessment.
        """
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        print(f"[TOOL] Error verifying information: {e}")
        return f"Error verifying information: {str(e)}"

# --- AGENT/CHAIN SETUP ---
# Combine all tools for a unified agent
all_tools = [
    Tool(
        name="search_papers",
        func=search_papers_tool,
        description=(
            "Search arXiv for academic papers on a given topic and return a list of papers. "
            "Use this tool when the user says things like: 'search', 'find papers', 'look up', 'get articles', 'show me papers about', 'list recent papers on', 'find research', or provides a research topic or keywords. "
            "Input: A research topic, keywords, or a general subject area. "
            "ONLY return the list of papers and STOP. "
            "Do NOT summarize, analyze, or answer questions unless explicitly asked."
        )
    ),
    Tool(
        name="rag",
        func=rag_tool,
        description=(
            "Answer questions about the last searched papers using Retrieval-Augmented Generation (RAG). "
            "Use this tool when the user asks a question about the content of the papers, such as: "
            "'what does the paper say about...', 'explain', 'analyze', 'details on', 'what is discussed in', 'what are the findings', 'what methods are used', 'what is the conclusion', or requests a summary/analysis. "
            "Input: A specific question about the content of the last searched papers. "
            "Use this tool ONLY if the user asks a question or requests a summary/analysis. "
            "Do NOT use this tool for trend analysis unless the user explicitly asks for trends."
        )
    ),
    Tool(
        name="summarize_paper",
        func=summarize_paper_tool,
        description=(
            "Download and summarize a specific paper. Use this tool when the user says things like: "
            "'summarize', 'give me a summary', 'what is this paper about', 'summarize this article', 'overview of', or provides an arXiv URL or clear reference to a paper in the context. "
            "Input: An arXiv URL or a clear reference to a specific paper. "
            "Use this tool ONLY if the user explicitly asks for a summary of a specific paper. "
            "Do NOT use this tool for trend analysis."
        )
    ),
    Tool(
        name="detect_trends",
        func=detect_trends_tool,
        description=(
            "Analyze papers to detect research trends and themes. Use this tool when the user says things like: "
            "'find trends', 'emerging topics', 'common themes', 'what are the trends', 'what is popular', 'what is new in', 'identify hot topics', or asks for an analysis of research directions. "
            "Input: The text or abstracts of the papers to analyze. "
            "Use this tool ONLY if the user explicitly asks for trends, themes, or emerging topics. "
            "Do NOT use this tool after a summary unless the user specifically requests trend analysis."
        )
    ),
    Tool(
        name="analyze_trends_in_topic",
        func=analyze_trends_in_topic,
        description=(
            "Search for papers on a topic and analyze trends in those papers. Use this tool when the user says things like: "
            "'analyze trends in', 'trend analysis', 'what are the trends in', 'what is the direction of research in', 'how is the field evolving', or requests a trend analysis for a topic. "
            "Input: A research topic or subject area. "
            "Use this tool ONLY if the user explicitly asks for a trend analysis. "
            "Do NOT use this tool after a summary unless the user specifically requests trend analysis."
        )
    ),
    Tool(
        name="verify_information",
        func=verify_information_tool,
        description=(
            "Verify the factual accuracy of a summary, answer, or claim against the full text of a paper. Use this tool when the user says things like: "
            "'verify', 'fact-check', 'is this correct', 'is this supported', 'is this true', 'can you check', 'is this claim accurate', or asks to check the accuracy of information. "
            "Input: The claim, summary, or answer to verify, and optionally an arXiv URL. "
            "Use this tool ONLY if the user explicitly asks to verify or fact-check information."
        )
    )
]

unified_agent = initialize_agent(
    tools=all_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # <--- Tool-only agent
    verbose=False,
    handle_parsing_errors=True,
    memory=memory,
    system_message=(
        "You are a research assistant with access to all tools: search, RAG, summarization, trend analysis, and verification. "
        "For each user request, you MUST call the most appropriate tool and return ONLY the tool's output as the final answer. "
        "NEVER answer from your own knowledge or memory for questions about paper content, summaries, or analysis—ALWAYS use the tools. "
        "Do NOT output any thoughts, reasoning, or intermediate steps."
    )
)

# --- MAIN LOOP ---
def main():
    print("🔬 Welcome! To search for papers, just type a topic. To ask a question about the papers, ask a follow-up after searching.")
    while True:
        user_query = input("\n💬 What would you like me to help you with? ").strip()
        if user_query.lower() in ['exit', 'quit', 'bye']:
            print("👋 Goodbye! Happy researching!")
            break
        if not user_query:
            continue
        # Provide context about last searched papers
        if last_searched_papers:
            context = get_last_searched_papers_context()
            full_input = f"{context}\n\n{user_query}"
        else:
            full_input = user_query

        # Let the unified agent decide which tool to use
        response = unified_agent.invoke({"input": full_input})
        print(f"\n📝 Response:\n{response['output']}")


if __name__ == "__main__":
    main()