from config import llm, ARXIV_MAX_RESULTS
from paper import Paper
from tools.search_tools import arxiv_search

def detect_trends_tool(papers_text: str) -> str:
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
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        return f"Error detecting trends: {str(e)}"

def analyze_trends_in_topic(topic: str) -> str:
    papers = arxiv_search(topic, ARXIV_MAX_RESULTS)
    if not papers:
        return f"No papers found for topic: {topic}"
    papers_text = f"Papers on '{topic}':\n\n"
    for i, paper in enumerate(papers, 1):
        papers_text += f"Paper {i}: {paper.title}\nAuthors: {paper.authors}\nAbstract: {paper.abstract}\n\n"
    trends = detect_trends_tool(papers_text)
    return f"Trend Analysis for '{topic}':\n\n{trends}" 