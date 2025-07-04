from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import os
import re
import fitz  # PyMuPDF
from collections import defaultdict
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from crewai import Agent, Task, Crew,  LLM
from pinecone import Pinecone
from io import BytesIO
import html
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import unicodedata
import asyncio
from typing import List
# ─── Load API Keys ───
load_dotenv()
API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ─── Init Pinecone ───
pc = Pinecone(api_key=API_KEY)
index_name = "quickstart-py1"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
    )
index = pc.Index(index_name)
embed_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=GOOGLE_API_KEY
)


def clean_chunk(chunk: str) -> str:
    # Filter out common junk patterns
    noise_patterns = [
        r'article history.*', r'available online.*', r'copyright.*',
        r'Keywords?:.*', r'journal homepage.*', r'©\s*\d{4}', r'doi:.*',
        r'www\..*', r'\.com', r'\.org', r'https?:\/\/\S+'
    ]
    for pattern in noise_patterns:
        chunk = re.sub(pattern, '', chunk, flags=re.IGNORECASE)
    return chunk.strip()


# Set up Gemini LLM
llm = LLM(model="gemini/gemini-1.5-flash", api_key=GOOGLE_API_KEY)


async def generate_agent_response(user_query: str, context_chunks: List[str]) -> str:
    cleaned_chunks = [clean_chunk(chunk)
                      for chunk in context_chunks if chunk.strip()]
    context = "\n\n".join(cleaned_chunks[:10])

    # Define Agent
    agent = Agent(
        name="PDF Intelligence Analyst",
        role="Advanced PDF Content Interpreter",
        goal=(
            "To assist users by accurately analyzing and extracting relevant insights from academic or medical PDFs. "
            "The agent ensures that every answer is grounded in the provided context, delivering clarity, factual accuracy, and relevance."
        ),
        backstory=(
            "You are a highly capable AI specialized in interpreting structured and unstructured data from PDF documents, "
            "particularly academic research, scientific papers, and medical literature. You are trained to locate and summarize "
            "the most relevant information, avoiding speculation or unsupported conclusions. "
            "Your responses are grounded in the content and avoid hallucination. "
            "You are helpful, logical, and precise. If the content is ambiguous, irrelevant, or insufficient, you must acknowledge that clearly. "
            "You are also capable of handling follow-up questions, drawing from prior context only when explicitly reloaded. "
            "You always respond concisely in 5–10 well-structured sentences, using bullet points if clarity can be improved. "
            "You do not include any personal opinions or assumptions. "
            "When appropriate, you refer directly to facts or sections of the PDF, without quoting excessively."
        ),
        llm=llm,
        verbose=True,
    )

    # Define Task
    task = Task(
        description=f"""
Context:
---------
{context}

Question:
---------
{user_query}

Instructions:
-------------
- Read the question carefully and locate the most relevant section in the context.
- Do **not** fabricate or assume any information not present in the context.
- If you **cannot** find a sufficient answer based on the context, respond with:
  `"The provided document does not contain enough information to answer this question accurately."`
- When answering, prefer clear and concise language (5–10 well-structured sentences).
- Where helpful, use numbered points or short bullet lists to improve readability.
- Do not quote large blocks from the PDF—summarize meaningfully instead.
- Focus on factual precision, especially when answering technical or scientific queries.

Your output should be informative, clear, and directly related to the user's query.
        """,
        expected_output="An informative, concise response (5–10 sentences), grounded in the context. No assumptions or hallucinations.",
        agent=agent,
    )

    # Run with Crew
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        llm=llm,
    )
    response = await asyncio.create_task(crew.kickoff_async())
    # result = crew.kickoff()  # synchronous call (or use await kickoff_async() if needed)
    return response

# ─── App Config ───
app = FastAPI(docs_url=None,        # disables Swagger UI at /docs
              redoc_url=None,       # disables ReDoc at /redoc
              openapi_url=None      # disables OpenAPI JSON at /openapi.json
              )
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ─── Models ───


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    book: str
    score: float
    text: str

# ─── Helper Functions ───


def contains_watermark_keyword(text, keywords):
    return any(re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE) for keyword in keywords)


def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if word not in ENGLISH_STOP_WORDS and len(word) > 2]


def parse_pdf_file(file_bytes: bytes):
    watermark_keywords = [
        "COPY", "WATERMARK", "CONFIDENTIAL", "DO NOT DISTRIBUTE",
        "PREVIEW", "DRAFT", "COPYRIGHT", "CONFIDENTIALITY", "FOR INTERNAL USE ONLY"
    ]
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    output_text = []
    for page in pdf:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block['type'] == 0:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span['text'].strip()
                        font_size = span.get('size', 0)
                        rotation = abs(span.get('rotation', 0))
                        opacity = span.get('opacity', 1)
                        color = span.get('color', None)

                        if (5 < font_size < 20 and rotation < 10 and opacity > 0.9
                                and color not in [8421504, 12632256, 0xCCCCCC]
                                and not contains_watermark_keyword(text, watermark_keywords)):
                            output_text.append(text)
    pdf.close()
    return "\n".join(output_text)


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
        doc = fitz.open(stream=BytesIO(content), filetype="pdf")

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


def sanitize_vector_id_title(title: str) -> str:
    # Normalize Unicode to ASCII-compatible
    ascii_title = unicodedata.normalize('NFKD', title).encode(
        'ascii', 'ignore').decode('ascii')
    # Replace non-alphanumerics with hyphens
    return re.sub(r'[^a-zA-Z0-9_-]+', '-', ascii_title).strip('-').lower()


def chunk_and_embed(text, book_title, filename):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(text)
    vectors = embed_model.embed_documents(chunks)
    safe_title = sanitize_vector_id_title(book_title)

    return [
        {
            "id": f"{safe_title}-chunk-{i}",
            "values": vector,
            "metadata": {
                "chunk_text": chunk,
                "book_title": book_title,
                "filename": filename,  # ✅ Add this line
                "chunk_index": i,
                "keywords": extract_keywords(chunk),
            }
        }
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]


def store_pdf_in_pinecone(file_bytes: bytes, book_title: str, filename: str):
    text = parse_pdf_file(file_bytes)
    records = chunk_and_embed(text, book_title, filename)
    index.upsert(vectors=records, namespace="example-namespace")


def enhance_prompt(user_query: str) -> str:
    query = user_query.strip()
    query_lower = query.lower()

    # Clean and standardize
    query = query[0].upper() + query[1:] if query else ""

    # Patterns that suggest already good prompts
    good_starters = ("elaborate", "explain", "describe",
                     "compare", "give", "provide", "what", "how", "why")
    if query_lower.startswith(good_starters):
        return query

    # Keywords to detect intent
    if len(query.split()) <= 3:
        # If too short, likely a topic
        return f"Explain the concept of {query} in detail with examples."

    # If query contains verbs like "use", "impact", "role", etc.
    if any(kw in query_lower for kw in ["impact", "importance", "role", "usage", "use", "application"]):
        return f"Discuss the {query_lower} in depth with real-world examples."

    # If it's a question without a question word
    if query.endswith("?"):
        return f"Answer the following question in detail: {query}"

    # Generic fallback
    return f"Explain in detail: {query}"


def extract_query_keywords(user_query: str) -> List[str]:
    keywords = extract_keywords(user_query)
    return keywords


def retrieve_query_results(user_query: str):
    enhanced_query = enhance_prompt(user_query)
    print(enhanced_query)
    query_vector = embed_model.embed_query(enhanced_query)
    print("Query vector generated")
    keywords = extract_query_keywords(enhanced_query)
    print("Keywords extracted:", keywords)
    results = index.query(
        vector=query_vector,
        top_k=25,
        namespace="example-namespace",
        include_metadata=True,
        # optional keyword filtering
        filter={"keywords": {"$in": keywords}}
    )
    print("Query executed")
    print("Query results:", results)

    # return results['matches']
    if all('keywords' in match.get('metadata', {}) for match in results.get('matches', [])):
        results['matches'] = rerank_by_keyword_overlap(results, keywords)
    return results['matches']


def retrieve_query_results_me(user_query: str, book_names: List[str]):
    if not book_names:
        print("No book names provided")
        return []
    enhanced_query = enhance_prompt(user_query)
    print("Enhanced query:", enhanced_query)

    query_vector = embed_model.embed_query(enhanced_query)
    print("Query vector generated")

    keywords = extract_query_keywords(enhanced_query)
    print("Keywords extracted:", keywords)

    # Build filter with book name constraint
    filter_condition = {
        "keywords": {"$in": keywords},
        "filename": {"$in": book_names}
    }

    results = index.query(
        vector=query_vector,
        top_k=25,
        namespace="example-namespace",
        include_metadata=True,
        filter=filter_condition
    )

    print("Query executed")
    print("Query results:", results)

    if all('keywords' in match.get('metadata', {}) for match in results.get('matches', [])):
        results['matches'] = rerank_by_keyword_overlap(results, keywords)

    return results['matches']


def rerank_by_keyword_overlap(results, query_keywords):
    def score(match):
        chunk_keywords = match.get('metadata', {}).get('keywords', [])
        return len(set(chunk_keywords) & set(query_keywords))

    matches = results.get('matches', [])
    print("Reranking matches based on keyword overlap")
    if not matches:
        return []
    return sorted(matches, key=score, reverse=True)


# ─── API Endpoints ───


@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    uploaded_titles = []
    print("uploading")
    for file in files:
        content = await file.read()
        # title = os.path.splitext(file.filename)[0]
        title = extract_pdf_title(content, file.filename)
        print(f"Processing file: {title}")
        store_pdf_in_pinecone(content, title, file.filename)
        uploaded_titles.append(title)
    return {"message": "✅ Files processed successfully", "uploaded_titles": uploaded_titles}

# @app.post("/query/", response_model=List[QueryResponse])


@app.post("/query/")
async def query_pdf(req: QueryRequest):
    matches = retrieve_query_results(req.question)
    print("Query started")
    if not matches:
        return JSONResponse(content={"message": "No data available", "results": []}, status_code=200)
    print("Query will return results")
    # Extract context for AI agent
    # Group chunks by book
    # Normalize scores
    max_score = max(match["score"] for match in matches) or 1e-6
    for match in matches:
        match["norm_score"] = match["score"] / max_score

    # Group by book
    book_chunks = defaultdict(list)
    book_scores = defaultdict(list)
    print("Grouping chunks by book")
    for match in matches:
        book = match["metadata"].get("book_title", "Unknown")
        chunk = match["metadata"].get("chunk_text", "")
        if chunk:
            book_chunks[book].append(chunk)
            book_scores[book].append(match["norm_score"])
    print(f"Found {len(book_chunks)} books with matching chunks")
    # Add AI agent result
    # responses = [
    #     QueryResponse(
    #         book=match["metadata"].get("book_title", "Unknown"),
    #         score=match["score"],
    #         text=clean_chunk(match["metadata"].get("chunk_text", ""))
    #     )
    #     for match in matches
    # ]

    # Add AI agent result at the beginning
    # responses.insert(0, QueryResponse(
    #     book="AI Agent",
    #     score=1.0,  # you can keep it highest or just use -1 if not used
    #     text=str(agent_output.raw)
    # ))

    # return responses

    book_responses = []

    for book, chunks in book_chunks.items():
        print(f"Processing book: {book} with {len(chunks)} chunks")
        agent_output = generate_agent_response(req.question, chunks)
        avg_score = sum(book_scores[book]) / len(book_scores[book])
        book_responses.append({
            "book": book,
            # Optional — or compute average match score
            "score": round(avg_score, 3),
            "text": str(agent_output.raw)
        })

    return JSONResponse(content={"results": book_responses}, status_code=200)

    # return {
    #     # ✅ Only return agent’s summarized paragraphPDF Content Analyzer
    #     "agent_response": str(agent_output.raw)
    # }


# @app.post("/query/", response_model=List[QueryResponse])
# async def query_pdf(req: QueryRequest):
#     matches = retrieve_query_results(req.question)
#     print(" quesry started")

#     if not matches:
#         return JSONResponse(content={"message": "No data available", "results": []}, status_code=200)
#     print(" quesry  will returnr started")
#     return [
#         QueryResponse(
#             book=match["metadata"].get("book_title", "Unknown"),
#             score=match["score"],
#             text=match["metadata"].get("chunk_text", "")
#         )
#         for match in matches
#     ]
class QueryMeRequest(BaseModel):
    question: str
    book_names: List[str]  # List of book/pdf names to filter on


@app.post("/queryme/")
async def query_pdf_with_filter(req: QueryMeRequest):
    print("QueryMe started")
    if not req.book_names:
        return JSONResponse(content={"message": "No book names provided", "results": []}, status_code=200)
    all_matches = retrieve_query_results_me(req.question, req.book_names)

    if not all_matches:
        return JSONResponse(content={"message": "No data available", "results": []}, status_code=200)

    # print(f"Filtering matches to specified books: {req.book_names}")
    # # Filter matches based on the list of book names
    # filtered_matches = [
    #     match for match in all_matches
    #     if match["metadata"].get("book_title", "").strip() in req.book_names
    # ]

    # Normalize scores
    max_score = max(match["score"] for match in all_matches) or 1e-6
    for match in all_matches:
        match["norm_score"] = match["score"] / max_score

    # Group by book
    book_chunks = defaultdict(list)
    book_scores = defaultdict(list)
    print("Grouping filtered chunks by book")
    for match in all_matches:
        book = match["metadata"].get("book_title", "Unknown")
        chunk = match["metadata"].get("chunk_text", "")
        if chunk:
            book_chunks[book].append(chunk)
            book_scores[book].append(match["norm_score"])

    print(f"Found {len(book_chunks)} books with matching chunks")

    # Build responses using AI agent
    book_responses = []
    for book, chunks in book_chunks.items():
        print(f"Processing book: {book} with {len(chunks)} chunks")
        agent_output = generate_agent_response(req.question, chunks)
        avg_score = sum(book_scores[book]) / len(book_scores[book])
        book_responses.append({
            "book": book,
            "score": round(avg_score, 3),
            "text": str(agent_output.raw)
        })

    return JSONResponse(content={"results": book_responses}, status_code=200)


@app.get("/")
def hello():
    return {"message": "Hello, this is the PDF Query API!"}
