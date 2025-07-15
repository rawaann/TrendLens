from config import llm
from paper import Paper, extract_arxiv_id

# Global cache for last searched papers
last_searched_papers = []

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
        return f"Error summarizing text: {str(e)}"

def summarize_paper_tool(paper_ref: str) -> str:
    # Try to find the Paper object by title, arXiv ID, or link
    matched_paper = None
    for paper in last_searched_papers:
        if paper_ref.lower() in paper.title.lower() or paper_ref in paper.link or paper_ref == paper.arxiv_id:
            matched_paper = paper
            break
    if not matched_paper:
        return "Paper not found in the last search. Please search for papers first."
    if not matched_paper.full_text:
        return "No full text available for summarization."
    summary = summarize_text_with_longt5(matched_paper.full_text, matched_paper.title, matched_paper.link)
    return f"Summary of {matched_paper.title} (arXiv: {matched_paper.arxiv_id}):\n\n{summary}" 