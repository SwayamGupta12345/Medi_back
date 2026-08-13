from google.api_core.exceptions import ResourceExhausted
import time
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from collections import defaultdict
from io import BytesIO
import os
import re
import html
import unicodedata
import pymupdf
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from crewai import Agent, Task, Crew, LLM
from pinecone import Pinecone
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import asyncio
# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is missing")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing")

# ============================================================
# PINECONE
# ============================================================

pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "medical-pdf-rag"
NAMESPACE = "example-namespace"
EMBEDDING_DIMENSION = 3072

if not pc.has_index(INDEX_NAME):
    print(f"Creating Pinecone index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    )

index = pc.Index(INDEX_NAME)

# ============================================================
# GOOGLE EMBEDDINGS
# ============================================================

embed_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# ============================================================
# GEMINI LLM
# ============================================================

llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=GOOGLE_API_KEY
)

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# MODELS
# ============================================================


class QueryRequest(BaseModel):
    question: str


class QueryMeRequest(BaseModel):
    question: str
    book_names: List[str]


class QueryResponse(BaseModel):
    book: str
    score: float
    text: str

# ============================================================
# TEXT CLEANING
# ============================================================


def clean_chunk(chunk: str) -> str:
    noise_patterns = [
        r"article history.*",
        r"available online.*",
        r"copyright.*",
        r"keywords?:.*",
        r"journal homepage.*",
        r"©\s*\d{4}",
        r"doi:.*",
        r"https?://\S+",
        r"www\.\S+",
    ]

    for pattern in noise_patterns:
        chunk = re.sub(pattern, "", chunk, flags=re.IGNORECASE)

    return chunk.strip()

# ============================================================
# KEYWORDS
# ============================================================


def extract_keywords(text: str) -> List[str]:
    words = re.findall(r"\b\w+\b", text.lower())

    return [
        word
        for word in words
        if word not in ENGLISH_STOP_WORDS and len(word) > 2
    ]

# ============================================================
# WATERMARK DETECTION
# ============================================================


def contains_watermark_keyword(text: str, keywords: List[str]) -> bool:
    return any(
        re.search(
            r"\b" + re.escape(keyword) + r"\b",
            text,
            re.IGNORECASE
        )
        for keyword in keywords
    )

# ============================================================
# PDF PARSING
# ============================================================


def parse_pdf_file(file_bytes: bytes) -> str:
    watermark_keywords = [
        "COPY",
        "WATERMARK",
        "CONFIDENTIAL",
        "DO NOT DISTRIBUTE",
        "PREVIEW",
        "DRAFT",
        "COPYRIGHT",
        "CONFIDENTIALITY",
        "FOR INTERNAL USE ONLY"
    ]

    pdf = pymupdf.open(stream=file_bytes, filetype="pdf")
    output_text = []

    try:
        for page in pdf:
            blocks = page.get_text("dict").get("blocks", [])

            for block in blocks:
                if block.get("type") != 0:
                    continue

                for line in block.get("lines", []):
                    line_text = []

                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()

                        if not text:
                            continue

                        if contains_watermark_keyword(
                            text,
                            watermark_keywords
                        ):
                            continue

                        line_text.append(text)

                    if line_text:
                        joined = " ".join(line_text).strip()

                        if joined:
                            output_text.append(joined)
    finally:
        pdf.close()

    text = "\n".join(output_text)
    return text.strip()


# ============================================================
# PDF TITLE EXTRACTION
# ============================================================

def extract_pdf_title(content: bytes, fallback_filename: str) -> str:
    def is_valid_title(text: str) -> bool:
        text = text.strip()
        if len(text) < 10:
            return False
        if text.lower() in {"untitled", "document", "new", "scan"}:
            return False
        if re.fullmatch(r"\d+", text):
            return False
        if not re.search(r"[a-zA-Z]", text):
            return False
        return True

    def is_author_line(text: str) -> bool:
        # Looks like list of names or affiliations
        return bool(re.search(r"\b(?:[A-Z]\w+\s+[A-Z]\w+|\d)\b", text)) and len(text) < 100

    def meaningful_word_count(text: str) -> int:
        words = re.findall(r'\b\w+\b', text.lower())
        return sum(1 for w in words if w not in ENGLISH_STOP_WORDS)

    try:
        doc = pymupdf.open(stream=BytesIO(content), filetype="pdf")

        # 1. Try metadata
        metadata_title = doc.metadata.get("title", "")
        if is_valid_title(metadata_title):
            return html.unescape(metadata_title.strip())

        # 2. Visual font-based scanning
        page = doc.load_page(0)
        blocks = page.get_text("dict")["blocks"]
        font_groups = defaultdict(list)

        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                line_text = ""
                font_sizes = []
                for span in line.get("spans", []):
                    txt = span.get("text", "").strip()
                    if txt:
                        line_text += txt + " "
                        font_sizes.append(span.get("size", 0))
                avg_font = sum(font_sizes) / \
                    len(font_sizes) if font_sizes else 0
                line_text = line_text.strip()
                if is_valid_title(line_text):
                    font_groups[round(avg_font, 1)].append(line_text)

        if not font_groups:
            raise ValueError("No valid font-based lines found")

        # Pick lines with largest font
        largest_font = max(font_groups.keys())
        candidates = font_groups[largest_font]

        # Filter and prioritize
        filtered = [
            (meaningful_word_count(text), ":" in text, text)
            for text in candidates
            if not is_author_line(text)
        ]
        if filtered:
            filtered.sort(reverse=True)
            return html.unescape(filtered[0][2].strip())

    except Exception as e:
        print(f"[extract_pdf_title] Error: {e}")

    return os.path.splitext(fallback_filename)[0]
# ============================================================
# VECTOR ID
# ============================================================


def sanitize_vector_id_title(title: str) -> str:
    ascii_title = (
        unicodedata
        .normalize("NFKD", title)
        .encode("ascii", "ignore")
        .decode("ascii")
    )

    cleaned = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "-",
        ascii_title
    )

    return cleaned.strip("-").lower()

# ============================================================
# CHUNK + EMBED
# ============================================================


# def chunk_and_embed(text: str, book_title: str, filename: str):
#     if not text.strip():
#         raise ValueError("PDF contains no extractable text")

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=75
#     )

#     chunks = splitter.split_text(text)

#     if not chunks:
#         raise ValueError("No chunks generated from PDF")

#     print(f"Generated {len(chunks)} chunks")

#     vectors = embed_model.embed_documents(chunks)

#     for i, vector in enumerate(vectors):
#         if len(vector) != EMBEDDING_DIMENSION:
#             raise ValueError(
#                 f"Embedding dimension mismatch at chunk {i}: "
#                 f"got {len(vector)}, expected {EMBEDDING_DIMENSION}"
#             )

#     safe_title = sanitize_vector_id_title(book_title)
#     records = []

#     for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
#         records.append({
#             "id": f"{safe_title}-chunk-{i}",
#             "values": vector,
#             "metadata": {
#                 "chunk_text": chunk,
#                 "book_title": book_title,
#                 "filename": filename,
#                 "chunk_index": i,
#                 "keywords": extract_keywords(chunk)
#             }
#         })

#     return records

# ============================================================
# CHUNK + EMBED (rate-limited)
# ============================================================


def embed_with_backoff(texts: List[str], max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            return embed_model.embed_documents(texts)
        except ResourceExhausted:
            wait = 2 ** attempt  # 1, 2, 4, 8, 16 sec
            print(f"Rate limited. Retrying in {wait}s (attempt {attempt+1})")
            time.sleep(wait)
    raise RuntimeError(
        "Embedding failed after max retries due to rate limiting")


def chunk_and_embed(text: str, book_title: str, filename: str):
    if not text.strip():
        raise ValueError("PDF contains no extractable text")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=75
    )

    chunks = splitter.split_text(text)

    if not chunks:
        raise ValueError("No chunks generated from PDF")

    print(f"Generated {len(chunks)} chunks")

    BATCH_SIZE = 20        # keep well under 100/min per batch
    SLEEP_BETWEEN_BATCHES = 2  # seconds

    vectors = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_vectors = embed_with_backoff(batch)
        vectors.extend(batch_vectors)

        if i + BATCH_SIZE < len(chunks):
            time.sleep(SLEEP_BETWEEN_BATCHES)

    for i, vector in enumerate(vectors):
        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding dimension mismatch at chunk {i}: "
                f"got {len(vector)}, expected {EMBEDDING_DIMENSION}"
            )

    safe_title = sanitize_vector_id_title(book_title)
    records = []

    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        records.append({
            "id": f"{safe_title}-chunk-{i}",
            "values": vector,
            "metadata": {
                "chunk_text": chunk,
                "book_title": book_title,
                "filename": filename,
                "chunk_index": i,
                "keywords": extract_keywords(chunk)
            }
        })

    return records

# ============================================================
# STORE PDF
# ============================================================


def store_pdf_in_pinecone(
    file_bytes: bytes,
    book_title: str,
    filename: str
):
    text = parse_pdf_file(file_bytes)

    print(f"Extracted text length: {len(text)}")

    if not text:
        raise ValueError("No text could be extracted from PDF")

    records = chunk_and_embed(
        text,
        book_title,
        filename
    )

    print(f"Upserting {len(records)} vectors")

    index.upsert(
        vectors=records,
        namespace=NAMESPACE
    )

    print("Pinecone upsert completed")

    return len(records)

# ============================================================
# QUERY ENHANCEMENT
# ============================================================


def enhance_prompt(user_query: str) -> str:
    query = user_query.strip()

    if not query:
        return ""

    query_lower = query.lower()

    good_starters = (
        "elaborate",
        "explain",
        "describe",
        "compare",
        "give",
        "provide",
        "what",
        "how",
        "why"
    )

    if query_lower.startswith(good_starters):
        return query

    if len(query.split()) <= 3:
        return (
            f"Explain the concept of "
            f"{query} in detail with examples."
        )

    if any(
        keyword in query_lower
        for keyword in [
            "impact",
            "importance",
            "role",
            "usage",
            "use",
            "application"
        ]
    ):
        return (
            f"Discuss the "
            f"{query_lower} "
            f"in depth with examples."
        )

    if query.endswith("?"):
        return (
            f"Answer the following question "
            f"in detail: {query}"
        )

    return f"Explain in detail: {query}"

# ============================================================
# QUERY KEYWORDS
# ============================================================


def extract_query_keywords(user_query: str) -> List[str]:
    return extract_keywords(user_query)

# ============================================================
# HYBRID RETRIEVAL
# ============================================================


def calculate_keyword_score(match, query_keywords):
    chunk_keywords = (
        match
        .get("metadata", {})
        .get("keywords", [])
    )

    query_set = set(query_keywords)
    chunk_set = set(chunk_keywords)

    if not query_set or not chunk_set:
        return 0.0

    intersection = query_set & chunk_set

    return len(intersection) / len(query_set)

# def hybrid_rerank(matches, query_keywords, alpha=0.7):
#     if not matches:
#         return []

#     for match in matches:
#         vector_score = float(
#             match.get("score", 0.0)
#         )

#         keyword_score = calculate_keyword_score(
#             match,
#             query_keywords
#         )

#         hybrid_score = (
#             alpha * vector_score
#             + (1 - alpha) * keyword_score
#         )

#         match["vector_score"] = round(
#             vector_score,
#             4
#         )

#         match["keyword_score"] = round(
#             keyword_score,
#             4
#         )

#         match["hybrid_score"] = round(
#             hybrid_score,
#             4
#         )

#     matches.sort(
#         key=lambda x: x["hybrid_score"],
#         reverse=True
#     )

#     return matches
# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================


def reciprocal_rank_fusion(matches, query_keywords, k_rrf=60):
    if not matches:
        return []

    # Rank list 1: dense vector score (already sorted by Pinecone, but re-sort to be safe)
    dense_ranked = sorted(
        matches, key=lambda m: m.get("score", 0.0), reverse=True)
    dense_rank = {id(m): rank for rank, m in enumerate(dense_ranked, start=1)}

    # Rank list 2: keyword overlap score (independent signal)
    def kw_score(m):
        return calculate_keyword_score(m, query_keywords)

    keyword_ranked = sorted(matches, key=kw_score, reverse=True)
    keyword_rank = {id(m): rank for rank,
                    m in enumerate(keyword_ranked, start=1)}

    for m in matches:
        r_dense = dense_rank[id(m)]
        r_kw = keyword_rank[id(m)]
        rrf_score = (1.0 / (k_rrf + r_dense)) + (1.0 / (k_rrf + r_kw))
        m["vector_score"] = round(float(m.get("score", 0.0)), 4)
        m["keyword_score"] = round(kw_score(m), 4)
        m["rrf_score"] = round(rrf_score, 6)

    matches.sort(key=lambda x: x["rrf_score"], reverse=True)
    return matches
# ============================================================
# RETRIEVE
# ============================================================

def match_to_dict(m):
    getter = m.get if hasattr(m, "get") else (lambda k, d=None: getattr(m, k, d))
    metadata = getter("metadata", {}) or {}
    return {
        "id": getter("id"),
        "score": getter("score", 0.0),
        "metadata": dict(metadata) if not isinstance(metadata, dict) else metadata,
    }
def retrieve_query_results(user_query: str):
    enhanced_query = enhance_prompt(user_query)

    print("Enhanced query:", enhanced_query)

    query_vector = embed_model.embed_query(
        enhanced_query
    )

    print(
        "Query vector generated:",
        len(query_vector)
    )

    if len(query_vector) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Query embedding dimension "
            f"{len(query_vector)} does not match "
            f"Pinecone index dimension "
            f"{EMBEDDING_DIMENSION}"
        )

    keywords = extract_query_keywords(
        enhanced_query
    )

    print("Keywords:", keywords)

    results = index.query(
        vector=query_vector,
        top_k=50,
        namespace=NAMESPACE,
        include_metadata=True
    )

    # matches = results.get("matches", [])
    matches = [match_to_dict(m) for m in results.get("matches", [])]

    print(
        "Dense matches:",
        len(matches)
    )

    matches = reciprocal_rank_fusion(
        matches,
        keywords
    )

    matches = matches[:20]

    print(
        "Final matches:",
        len(matches)
    )

    for match in matches[:5]:
        print(
            "MATCH:",
            match.get("id"),
            match.get("score"),
            match.get("rrf_score"),
            match.get("metadata", {}).get("book_title")
        )

    return matches

# ============================================================
# RETRIEVE FROM SPECIFIC BOOKS
# ============================================================


def retrieve_query_results_me(
    user_query: str,
    book_names: List[str]
):
    if not book_names:
        return []

    enhanced_query = enhance_prompt(user_query)

    print(
        "Enhanced query:",
        enhanced_query
    )

    query_vector = embed_model.embed_query(
        enhanced_query
    )

    if len(query_vector) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Query embedding dimension "
            f"{len(query_vector)} does not match "
            f"index dimension "
            f"{EMBEDDING_DIMENSION}"
        )

    keywords = extract_query_keywords(
        enhanced_query
    )

    filter_condition = {
        "filename": {
            "$in": book_names
        }
    }

    results = index.query(
        vector=query_vector,
        top_k=50,
        namespace=NAMESPACE,
        include_metadata=True,
        filter=filter_condition
    )

    # matches = results.get("matches", [])
    matches = [match_to_dict(m) for m in results.get("matches", [])]

    print(
        "Filtered dense matches:",
        len(matches)
    )

    matches = reciprocal_rank_fusion(
        matches,
        keywords
    )
    for match in matches[:5]:
        print(
            "MATCH:",
            match.get("id"),
            match.get("score"),
            match.get("rrf_score"),
            match.get("metadata", {}).get("book_title")
        )

    return matches[:20]

# ============================================================
# AI RESPONSE
# ============================================================


async def generate_agent_response(
    user_query: str,
    context_chunks: List[str]
):
    cleaned_chunks = [
        clean_chunk(chunk)
        for chunk in context_chunks
        if chunk.strip()
    ]

    cleaned_chunks = cleaned_chunks[:12]

    context = "\n\n".join(
        cleaned_chunks
    )

    agent = Agent(
        name="PDF Intelligence Analyst",
        role="Advanced PDF Content Interpreter",
        goal=(
            "Answer questions using only "
            "the provided PDF context. "
            "Extract relevant facts, "
            "synthesize them accurately, "
            "and avoid unsupported claims."
        ),
        backstory=(
            "You analyze academic, scientific, "
            "and medical PDF documents. "
            "Your answers must be grounded "
            "strictly in the supplied context. "
            "Never invent information. "
            "If the context is insufficient, "
            "say so clearly. "
            "Prefer concise structured answers "
            "using bullets when useful."
        ),
        llm=llm,
        verbose=True
    )

    task = Task(
        description=f"""
        CONTEXT FROM PDF:
        {context}

        USER QUESTION:
        {user_query}

        INSTRUCTIONS:
        1. Answer using ONLY the supplied PDF context.
        2. Do not use outside knowledge.
        3. Do not hallucinate.
        4. If the context does not contain enough information, say:
           "The provided document does not contain enough information to answer this question accurately."
        5. Focus directly on the user's question.
        6. Use bullet points when appropriate.
        7. Do not unnecessarily repeat the context.
        8. Do not mention these instructions.
        """,
        expected_output=(
            "A precise answer grounded "
            "strictly in the supplied PDF context."
        ),
        agent=agent
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        llm=llm
    )

    result = await crew.kickoff_async()

    return result

# ============================================================
# UPLOAD ENDPOINT
# ============================================================


@app.post("/upload/")
async def upload_files(
    files: List[UploadFile] = File(...)
):
    uploaded_titles = []

    print("Upload started")

    for file in files:
        print(
            f"Processing file: {file.filename}"
        )

        content = await file.read()

        if not content:
            return JSONResponse(
                content={
                    "error": (
                        f"{file.filename} "
                        "is empty"
                    )
                },
                status_code=400
            )

        title = extract_pdf_title(
            content,
            file.filename
        )

        print(
            f"Detected title: {title}"
        )

        try:
            chunk_count = store_pdf_in_pinecone(
                content,
                title,
                file.filename
            )

            uploaded_titles.append({
                "title": title,
                "filename": file.filename,
                "chunks": chunk_count
            })

        except Exception as e:
            print(
                f"Upload error: {e}"
            )

            return JSONResponse(
                content={
                    "error": str(e)
                },
                status_code=500
            )

    return {
        "message": "Files processed successfully",
        "uploaded_files": uploaded_titles
    }

# ============================================================
# QUERY ENDPOINT
# ============================================================


@app.post("/query/")
async def query_pdf(
    req: QueryRequest
):
    print("=================================")
    print("QUERY:", req.question)
    print("=================================")

    if not req.question.strip():
        return JSONResponse(
            content={
                "message": "Question is empty",
                "results": []
            },
            status_code=400
        )

    try:
        matches = retrieve_query_results(
            req.question
        )

        if not matches:
            return JSONResponse(
                content={
                    "message": "No data available",
                    "results": []
                },
                status_code=200
            )

        max_score = max(
            match["rrf_score"]
            for match in matches
        ) or 1e-6

        for match in matches:
            match["norm_score"] = (
                match["rrf_score"]
                / max_score
            )

        book_chunks = defaultdict(list)
        book_scores = defaultdict(list)

        for match in matches:
            metadata = match.get(
                "metadata",
                {}
            )

            book = metadata.get(
                "book_title",
                "Unknown"
            )

            chunk = metadata.get(
                "chunk_text",
                ""
            )

            if chunk:
                book_chunks[book].append(chunk)
                book_scores[book].append(
                    match["norm_score"]
                )

        print(
            "Books found:",
            list(book_chunks.keys())
        )

        book_responses = []

        for book, chunks in book_chunks.items():
            print(
                f"Generating answer for {book}"
            )

            agent_output = await generate_agent_response(
                req.question,
                chunks
            )

            avg_score = (
                sum(book_scores[book])
                / len(book_scores[book])
            )

            book_responses.append({
                "book": book,
                "score": round(avg_score, 3),
                "text": str(agent_output)
            })

        return JSONResponse(
            content={
                "results": book_responses
            },
            status_code=200
        )

    except Exception as e:
        print(
            "QUERY ERROR:",
            repr(e)
        )

        return JSONResponse(
            content={
                "error": str(e),
                "error_type": type(e).__name__
            },
            status_code=500
        )

# ============================================================
# QUERY SPECIFIC PDFs
# ============================================================


@app.post("/queryme/")
async def query_pdf_with_filter(
    req: QueryMeRequest
):
    print("QueryMe started")

    if not req.book_names:
        return JSONResponse(
            content={
                "message": "No book names provided",
                "results": []
            },
            status_code=400
        )

    try:
        matches = retrieve_query_results_me(
            req.question,
            req.book_names
        )

        if not matches:
            return JSONResponse(
                content={
                    "message": "No data available",
                    "results": []
                },
                status_code=200
            )

        max_score = max(
            match["rrf_score"]
            for match in matches
        ) or 1e-6

        for match in matches:
            match["norm_score"] = (
                match["rrf_score"]
                / max_score
            )

        book_chunks = defaultdict(list)
        book_scores = defaultdict(list)

        for match in matches:
            metadata = match.get(
                "metadata",
                {}
            )

            book = metadata.get(
                "book_title",
                "Unknown"
            )

            chunk = metadata.get(
                "chunk_text",
                ""
            )

            if chunk:
                book_chunks[book].append(chunk)
                book_scores[book].append(
                    match["norm_score"]
                )

        book_responses = []

        for book, chunks in book_chunks.items():
            agent_output = await generate_agent_response(
                req.question,
                chunks
            )

            avg_score = (
                sum(book_scores[book])
                / len(book_scores[book])
            )

            book_responses.append({
                "book": book,
                "score": round(avg_score, 3),
                "text": str(agent_output)
            })

        return JSONResponse(
            content={
                "results": book_responses
            },
            status_code=200
        )

    except Exception as e:
        print(
            "QUERYME ERROR:",
            repr(e)
        )

        return JSONResponse(
            content={
                "error": str(e),
                "error_type": type(e).__name__
            },
            status_code=500
        )

# ============================================================
# HEALTH CHECK
# ============================================================


@app.get("/")
def hello():
    return {
        "message": "Hello, this is the PDF Query API!"
    }


@app.head("/")
def head_root():
    return Response(
        status_code=200
    )
