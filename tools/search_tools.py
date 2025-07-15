from config import ARXIV_MAX_RESULTS
from paper import Paper, extract_arxiv_id
import requests
import lxml.etree as etree
# Import the global cache from summarization_tools
from tools.summarization_tools import last_searched_papers

def arxiv_search(user_query, max_results=ARXIV_MAX_RESULTS):
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
            results.append(Paper(
                title=title_text,
                authors=authors_str,
                abstract=summary_text[:300] + "..." if len(summary_text) > 300 else summary_text,
                link=link
            ))
        return results
    except Exception:
        return []

def search_papers_tool(topic: str) -> str:
    papers = arxiv_search(topic, ARXIV_MAX_RESULTS)
    # Download and extract full text for each paper
    for paper in papers:
        paper.download_pdf()
        paper.extract_text()
    last_searched_papers.clear()
    last_searched_papers.extend(papers)
    if not papers:
        return f"No papers found for topic: {topic}"
    result = f"Found {len(papers)} papers for '{topic}':\n\n"
    for i, paper in enumerate(papers, 1):
        result += f"{i}. {paper.title}\n   Authors: {paper.authors}\n   Link: {paper.link}\n   Abstract: {paper.abstract}\n\n"
    return result 