from config import llm
from paper import Paper, extract_arxiv_id

def verify_information_tool(claim: str, paper: Paper = None) -> str:
    try:
        if paper is None or not paper.full_text:
            return "No paper or full text available for verification."
        all_text = paper.full_text
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
        return f"Error verifying information: {str(e)}" 