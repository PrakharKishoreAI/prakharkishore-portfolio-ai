import os
import re
import math
from collections import Counter

from groq import Groq


# ============================================================
# SETTINGS
# ============================================================

RESUME_PATH = "data/resume.txt"
PORTFOLIO_PATH = "data/portfolio.txt"

GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5


# ============================================================
# GROQ
# ============================================================

groq_api_key = os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found.")

client = Groq(api_key=groq_api_key)


# ============================================================
# READ TEXT FILE
# ============================================================

def read_text_file(path):
    if not os.path.exists(path):
        print(f"WARNING: File not found: {path}")
        return ""

    print(f"Reading file: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return file.read()


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(text, source, chunk_size=700, overlap=100):

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append({
                "text": chunk.strip(),
                "source": source
            })

        start += chunk_size - overlap

    return chunks


# ============================================================
# LIGHTWEIGHT TOKENIZER
# ============================================================

def tokenize(text):

    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    stopwords = {
        "the", "a", "an", "and", "or", "is",
        "are", "was", "were", "to", "of", "in",
        "on", "for", "with", "my", "his", "her",
        "this", "that", "it", "as", "at", "by",
        "from", "be", "has", "have", "had"
    }

    return [
        word
        for word in words
        if word not in stopwords and len(word) > 1
    ]


# ============================================================
# LIGHTWEIGHT TF-IDF RETRIEVER
# ============================================================

class LightweightRetriever:

    def __init__(self, chunks):

        self.chunks = chunks
        self.documents = []

        for chunk in chunks:
            self.documents.append(
                tokenize(chunk["text"])
            )

        self.document_count = len(self.documents)

        self.idf = {}

        document_frequency = Counter()

        for document in self.documents:

            unique_words = set(document)

            for word in unique_words:
                document_frequency[word] += 1

        for word, frequency in document_frequency.items():

            self.idf[word] = math.log(
                (self.document_count + 1)
                / (frequency + 1)
            ) + 1

        self.vectors = []

        for document in self.documents:
            self.vectors.append(
                self.create_vector(document)
            )

    def create_vector(self, words):

        counts = Counter(words)

        total = len(words)

        if total == 0:
            return {}

        vector = {}

        for word, count in counts.items():

            if word not in self.idf:
                continue

            tf = count / total

            vector[word] = tf * self.idf[word]

        return vector

    def similarity(self, vector_a, vector_b):

        if not vector_a or not vector_b:
            return 0.0

        score = 0.0

        for word, value in vector_a.items():

            if word in vector_b:
                score += value * vector_b[word]

        magnitude_a = math.sqrt(
            sum(value * value for value in vector_a.values())
        )

        magnitude_b = math.sqrt(
            sum(value * value for value in vector_b.values())
        )

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return score / (magnitude_a * magnitude_b)

    def search(self, query, top_k=5):

        query_words = tokenize(query)

        query_vector = self.create_vector(query_words)

        scores = []

        for index, document_vector in enumerate(self.vectors):

            score = self.similarity(
                query_vector,
                document_vector
            )

            scores.append((score, index))

        scores.sort(
            key=lambda item: item[0],
            reverse=True
        )

        results = []

        for score, index in scores[:top_k]:

            results.append({
                "text": self.chunks[index]["text"],
                "source": self.chunks[index]["source"],
                "score": float(score)
            })

        return results


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

print("\n" + "=" * 60)
print("LOADING KNOWLEDGE BASE")
print("=" * 60)

resume_text = read_text_file(RESUME_PATH)
portfolio_text = read_text_file(PORTFOLIO_PATH)

print("Resume characters:", len(resume_text))
print("Portfolio characters:", len(portfolio_text))


# ============================================================
# CREATE CHUNKS
# ============================================================

resume_chunks = create_chunks(
    resume_text,
    "Resume TXT"
)

portfolio_chunks = create_chunks(
    portfolio_text,
    "Portfolio TXT"
)

chunks = resume_chunks + portfolio_chunks

print("Resume chunks:", len(resume_chunks))
print("Portfolio chunks:", len(portfolio_chunks))
print("Total chunks:", len(chunks))


if not chunks:
    raise ValueError(
        "No text was found in resume.txt or portfolio.txt."
    )


# ============================================================
# CREATE LIGHTWEIGHT RETRIEVER
# ============================================================

print("\n" + "=" * 60)
print("CREATING LIGHTWEIGHT RAG RETRIEVER")
print("=" * 60)

retriever = LightweightRetriever(chunks)

print("Retriever created successfully.")
print("Total chunks:", len(chunks))


# ============================================================
# SEARCH
# ============================================================

def search(query, top_k=TOP_K):

    if not query or not query.strip():
        return []

    return retriever.search(
        query,
        top_k=top_k
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, results):

    if not results:
        return (
            "I couldn't find that information in "
            "Prakhar's portfolio."
        )

    evidence_parts = []

    for i, result in enumerate(results, start=1):

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
   say:
   "I couldn't find that information in Prakhar's portfolio."
4. Keep the answer clear, natural and concise.
5. Do not mention the retrieval process unless asked.
6. Do not confuse personal projects with academic major projects.
7. If the question is about education, work experience,
   projects, skills, certifications or achievements,
   use the Resume TXT and Portfolio TXT evidence.
8. When useful, combine information from both sources.

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

    return response.choices[0].message.content.strip()


# ============================================================
# SOURCES
# ============================================================

def get_sources(results):

    sources = []

    for result in results:

        source = result["source"]

        if source not in sources:
            sources.append(source)

    return sources


# ============================================================
# READY
# ============================================================

print("\n" + "=" * 60)
print("RAG PIPELINE READY")
print("=" * 60)

print("Sources:")
print("1. Resume TXT")
print("2. Portfolio TXT")
print("3. Lightweight TF-IDF Retrieval")
print("4. Groq Llama 3.3 70B")
print("=" * 60)