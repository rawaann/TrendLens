from config import llm
from paper import Paper
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from tools.summarization_tools import last_searched_papers

PDF_CHUNK_SIZE = 500
PDF_CHUNK_OVERLAP = 100

# These should be managed at a higher level, but for now, keep as module-level state
last_vectorstore = None
last_chunk_to_paper = None

# Helper to build vectorstore from papers

def build_vectorstore_from_papers(papers, embedding_model=None):
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
        full_text = getattr(paper, 'full_text', '')
        if not full_text:
            continue
        chunks = text_splitter.split_text(full_text)
        for chunk_idx, chunk in enumerate(chunks):
            all_documents.append(Document(
                page_content=chunk,
                metadata={
                    "paper_title": getattr(paper, 'title', 'Unknown Title'),
                    "arxiv_id": getattr(paper, 'arxiv_id', 'Unknown'),
                    "chunk_idx": chunk_idx,
                    "paper_idx": idx
                }
            ))
            chunk_to_paper.append(paper)
    if not all_documents:
        return None, None
    vectorstore = FAISS.from_documents(all_documents, embedding_model)
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

def rag_tool(question: str) -> str:
    global last_vectorstore, last_chunk_to_paper
    papers = last_searched_papers
    if not papers:
        return "No papers available. Please search for papers first."
    # Use cached vectorstore if available
    vectorstore = last_vectorstore
    chunk_to_paper = last_chunk_to_paper
    if not vectorstore or not chunk_to_paper:
        vectorstore, chunk_to_paper = build_vectorstore_from_papers(papers)
        last_vectorstore, last_chunk_to_paper = vectorstore, chunk_to_paper
    if not vectorstore:
        return "No full text available for RAG. Try searching for different papers."
    answer, sources = rag_qa(vectorstore, question)
    return answer 