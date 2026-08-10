# ============================================================
# Prakhar Portfolio AI - RAG Service
# Resume PDF + Portfolio TXT + FAISS + Groq
# With source-aware retrieval
# ============================================================

import os
import fitz
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# 1. SETTINGS
# ============================================================

PDF_PATH = "data/resume.pdf"
PORTFOLIO_PATH = "data/portfolio.txt"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

TOP_K = 5


# ============================================================
# 2. GROQ API
# ============================================================

groq_api_key = os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY not found. "
        "Please set your Groq API key in the terminal."
    )

client = Groq(api_key=groq_api_key)


# ============================================================
# 3. LOAD EMBEDDING MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING EMBEDDING MODEL")
print("=" * 60)

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Embedding model loaded successfully.")


# ============================================================
# 4. READ RESUME PDF
# ============================================================

def read_pdf(path):
    """
    Extract text from resume PDF.
    """

    if not os.path.exists(path):
        print(f"WARNING: PDF not found: {path}")
        return ""

    print(f"Reading PDF: {path}")

    document = fitz.open(path)

    text = ""

    for page in document:
        text += page.get_text() + "\n"

    document.close()

    return text


# ============================================================
# 5. READ PORTFOLIO TXT
# ============================================================

def read_portfolio(path):
    """
    Read portfolio information from portfolio.txt.
    """

    if not os.path.exists(path):
        print(f"WARNING: Portfolio file not found: {path}")
        return ""

    print(f"Reading portfolio: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return file.read()


# ============================================================
# 6. TEXT CHUNKING
# ============================================================

def create_chunks(text, source, chunk_size=700, overlap=100):
    """
    Split text into smaller chunks and keep source information.
    """

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(
                {
                    "text": chunk.strip(),
                    "source": source
                }
            )

        start += chunk_size - overlap

    return chunks


# ============================================================
# 7. LOAD KNOWLEDGE BASE
# ============================================================

print("\n" + "=" * 60)
print("LOADING KNOWLEDGE BASE")
print("=" * 60)

resume_text = read_pdf(PDF_PATH)
portfolio_text = read_portfolio(PORTFOLIO_PATH)

print("Resume characters:", len(resume_text))
print("Portfolio characters:", len(portfolio_text))


# ============================================================
# 8. CREATE SOURCE-AWARE CHUNKS
# ============================================================

resume_chunks = create_chunks(
    resume_text,
    "Resume PDF"
)

portfolio_chunks = create_chunks(
    portfolio_text,
    "Portfolio TXT"
)

chunks = resume_chunks + portfolio_chunks

print("\nNumber of chunks:", len(chunks))


# ============================================================
# 9. CREATE EMBEDDINGS
# ============================================================

print("\n" + "=" * 60)
print("CREATING EMBEDDINGS")
print("=" * 60)

if not chunks:
    raise ValueError(
        "No text was found in resume.pdf or portfolio.txt."
    )


chunk_texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = embedding_model.encode(
    chunk_texts,
    convert_to_numpy=True,
    show_progress_bar=True
)

embeddings = np.asarray(
    embeddings,
    dtype="float32"
)

print("Number of embeddings:", len(embeddings))
print("Embedding size:", embeddings.shape[1])


# ============================================================
# 10. CREATE FAISS INDEX
# ============================================================

print("\n" + "=" * 60)
print("CREATING FAISS INDEX")
print("=" * 60)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("FAISS index created successfully.")
print("Total vectors:", index.ntotal)


# ============================================================
# 11. SEARCH RELEVANT INFORMATION
# ============================================================

def search(query, top_k=TOP_K):
    """
    Search the knowledge base using semantic similarity.
    Returns retrieved text + source + similarity distance.
    """

    if not query or not query.strip():
        return []

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    distances, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx < 0 or idx >= len(chunks):
            continue

        results.append(
            {
                "text": chunks[idx]["text"],
                "source": chunks[idx]["source"],
                "score": float(distance)
            }
        )

    return results


# ============================================================
# 12. GENERATE ANSWER USING GROQ
# ============================================================

def generate_answer(question, results):
    """
    Generate an answer using only retrieved evidence.
    """

    if not results:
        return (
            "I couldn't find that information in "
            "Prakhar's portfolio."
        )

    evidence_parts = []

    for i, result in enumerate(
        results,
        start=1
    ):

        evidence_parts.append(
            f"""
EVIDENCE {i}
SOURCE: {result["source"]}

{result["text"]}
"""
        )

    evidence = "\n".join(evidence_parts)

    prompt = f"""
You are Prakhar's Portfolio AI assistant.

Answer the user's question using ONLY the evidence
provided below.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the information is not present in the evidence,
   say exactly:
   "I couldn't find that information in Prakhar's portfolio."
4. Keep the answer clear, natural and concise.
5. Do not mention the retrieval process unless asked.
6. Do not confuse personal projects with academic
   major projects.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{evidence}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful portfolio assistant. "
                    "Use only the provided evidence."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=500
    )

    answer = response.choices[0].message.content

    return answer.strip()


# ============================================================
# 13. GET SOURCE INFORMATION
# ============================================================

def get_sources(results):
    """
    Return unique sources used for the answer.
    """

    sources = []

    for result in results:

        source = result["source"]

        if source not in sources:
            sources.append(source)

    return sources


# ============================================================
# 14. STARTUP INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("RAG PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Sources:")
print("1. Resume PDF")
print("2. Portfolio TXT")
print("3. Sentence Transformer")
print("4. FAISS")
print("5. Groq Llama 3.3 70B")

print("=" * 60)