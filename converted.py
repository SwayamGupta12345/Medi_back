# from fastapi import FastAPI, UploadFile, File, Response
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from typing import List
# import os
# import re
# import fitz  # PyMuPDF
# from collections import defaultdict
# from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from crewai import Agent, Task, Crew,  LLM
# from pinecone import Pinecone
# from io import BytesIO
# import html
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# import unicodedata
# import asyncio
# from typing import List
# # ─── Load API Keys ───
# load_dotenv()
# API_KEY = os.getenv("PINECONE_API_KEY")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import os
import re
import fitz
from collections import defaultdict
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

# # ─── Init Pinecone ───
# pc = Pinecone(api_key=API_KEY)
# # index_name = "quickstart-py1"
# # if not pc.has_index(index_name):
# #     pc.create_index(
# #         name=index_name,
# #         dimension=768,
# #         metric="cosine",
# #         spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
# #     )
# # index = pc.Index(index_name)
# # embed_model = GoogleGenerativeAIEmbeddings(
# #     model="models/embedding-001", google_api_key=GOOGLE_API_KEY
# # )
# index_name = "medical-pdf-rag"

# if not pc.has_index(index_name):
#     pc.create_index(
#         name=index_name,
#         dimension=3072,
#         metric="cosine",
#         spec={
#             "serverless": {
#                 "cloud": "aws",
#                 "region": "us-east-1"
#             }
#         }
#     )

# index = pc.Index(index_name)
# # embed_model = GoogleGenerativeAIEmbeddings(
# #     model="models/embedding-001", google_api_key=GOOGLE_API_KEY
# # )

# embed_model = GoogleGenerativeAIEmbeddings(
#     model="models/gemini-embedding-001",
#     google_api_key=GOOGLE_API_KEY
# )


# def clean_chunk(chunk: str) -> str:
#     # Filter out common junk patterns
#     noise_patterns = [
#         r'article history.*', r'available online.*', r'copyright.*',
#         r'Keywords?:.*', r'journal homepage.*', r'©\s*\d{4}', r'doi:.*',
#         r'www\..*', r'\.com', r'\.org', r'https?:\/\/\S+'
#     ]
#     for pattern in noise_patterns:
#         chunk = re.sub(pattern, '', chunk, flags=re.IGNORECASE)
#     return chunk.strip()


# # Set up Gemini LLM
# llm = LLM(model="gemini/gemini-1.5-flash", api_key=GOOGLE_API_KEY)


# # async def generate_agent_response(user_query: str, context_chunks: List[str]) -> str:
# #     cleaned_chunks = [clean_chunk(chunk)
# #                       for chunk in context_chunks if chunk.strip()]
# #     context = "\n\n".join(cleaned_chunks[:10])

# #     # Define Agent
# #     agent = Agent(
# #         name="PDF Intelligence Analyst",
# #         role="Advanced PDF Content Interpreter",
# #         goal=(
# #             "To assist users by accurately analyzing and extracting relevant insights from academic or medical PDFs. "
# #             "The agent ensures that every answer is grounded in the provided context, delivering clarity, factual accuracy, and relevance."
# #         ),
# #         backstory=(
# #             "You are a highly capable AI specialized in interpreting structured and unstructured data from PDF documents, "
# #             "particularly academic research, scientific papers, and medical literature. You are trained to locate and summarize "
# #             "the most relevant information, avoiding speculation or unsupported conclusions. "
# #             "Your responses are grounded in the content and avoid hallucination. "
# #             "You are helpful, logical, and precise. If the content is ambiguous, irrelevant, or insufficient, you must acknowledge that clearly. "
# #             "You are also capable of handling follow-up questions, drawing from prior context only when explicitly reloaded. "
# #             "You always respond concisely in 5–10 well-structured sentences, using bullet points if clarity can be improved. "
# #             "You do not include any personal opinions or assumptions. "
# #             "When appropriate, you refer directly to facts or sections of the PDF, without quoting excessively."
# #         ),
# #         llm=llm,
# #         verbose=True,
# #     )

# #     # Define Task
# #     task = Task(
# #         description=f"""
# # Context:
# # ---------
# # {context}

# # Question:
# # ---------
# # {user_query}

# # Instructions:
# # -------------
# # - Read the question carefully and locate the most relevant section in the context.
# # - Do **not** fabricate or assume any information not present in the context.
# # - If you **cannot** find a sufficient answer based on the context, respond with:
# #   `"The provided document does not contain enough information to answer this question accurately."`
# # - When answering, prefer clear and concise language (5–10 well-structured sentences).
# # - Where helpful, use numbered points or short bullet lists to improve readability.
# # - Do not quote large blocks from the PDF—summarize meaningfully instead.
# # - Focus on factual precision, especially when answering technical or scientific queries.

# # Your output should be informative, clear, and directly related to the user's query.
# #         """,
# #         expected_output="An informative, concise response (5–10 sentences), grounded in the context. No assumptions or hallucinations.",
# #         agent=agent,
# #     )

# #     # Run with Crew
# #     crew = Crew(
# #         agents=[agent],
# #         tasks=[task],
# #         verbose=True,
# #         llm=llm,
# #     )
# #     response = await asyncio.create_task(crew.kickoff_async())
# #     # result = crew.kickoff()  # synchronous call (or use await kickoff_async() if needed)
# #     return response
# async def generate_agent_response(user_query: str, context_chunks: List[str]) -> str:
#     cleaned_chunks = [clean_chunk(chunk)
#                       for chunk in context_chunks if chunk.strip()]
#     # Increased from 10 to 20 for richer context
#     context = "\n\n".join(cleaned_chunks[:20])

#     # Define Agent
#     agent = Agent(
#         name="PDF Intelligence Analyst",
#         role="Advanced PDF Content Interpreter",
#         goal=(
#             "To extract, synthesize, and present comprehensive insights from academic or medical PDFs. "
#             "Every answer must be grounded in the provided context, prioritizing clarity, depth, and factual accuracy."
#         ),
#         backstory=(
#             "You are a highly capable AI trained to interpret complex academic and medical literature. "
#             "Your expertise lies in reading structured/unstructured content, understanding nuances, and delivering insightful, well-structured summaries. "
#             "You never hallucinate or speculate. You are analytical, concise, and context-driven. "
#             "You can handle difficult follow-up questions and clarify when data is insufficient. "
#             "When clarity can be improved, use bullet points or numbered structures. "
#             "You respond in 8–15 well-structured sentences or fewer if the answer is simple, and you avoid unnecessary repetition or filler."
#         ),
#         llm=llm,
#         verbose=True,
#     )

#     # Define Task
#     task = Task(
#         description=f"""
# Context:
# ---------
# {context}

# Question:
# ---------
# {user_query}

# Instructions:
# -------------
# - Analyze the context and locate the **most relevant** information.
# - Do **not** guess, fabricate, or include unsupported claims.
# - If there's **not enough context**, respond with:
#   `"The provided document does not contain enough information to answer this question accurately."`
# - Provide a **comprehensive and logically structured answer**:
#   - Prefer 8–15 well-formed sentences (unless answer is short by nature).
#   - Use **bullet points or numbered steps** for technical clarity.
#   - Reference specific sections or data patterns *only when helpful*.
# - Avoid quoting large chunks — paraphrase meaningfully and precisely.
# - Be informative, clear, technical when required, and avoid personal tone or assumptions.

# Respond as a domain expert—confident, precise, and insightful.
#         """,
#         expected_output="A comprehensive, precise answer grounded in the provided context, ideally 8–15 sentences or structured bullets.",
#         agent=agent,
#     )

#     # Run with Crew
#     crew = Crew(
#         agents=[agent],
#         tasks=[task],
#         verbose=True,
#         llm=llm,
#     )
#     response = await asyncio.create_task(crew.kickoff_async())
#     return response

# # ─── App Config ───
# app = FastAPI(docs_url=None,        # disables Swagger UI at /docs
#               redoc_url=None,       # disables ReDoc at /redoc
#               openapi_url=None      # disables OpenAPI JSON at /openapi.json
#               )
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Adjust for production
#     allow_credentials=False,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# # ─── Models ───


# class QueryRequest(BaseModel):
#     question: str


# class QueryResponse(BaseModel):
#     book: str
#     score: float
#     text: str

# # ─── Helper Functions ───


# def contains_watermark_keyword(text, keywords):
#     return any(re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE) for keyword in keywords)


# def extract_keywords(text):
#     words = re.findall(r'\b\w+\b', text.lower())
#     return [word for word in words if word not in ENGLISH_STOP_WORDS and len(word) > 2]


# def parse_pdf_file(file_bytes: bytes):
#     watermark_keywords = [
#         "COPY", "WATERMARK", "CONFIDENTIAL", "DO NOT DISTRIBUTE",
#         "PREVIEW", "DRAFT", "COPYRIGHT", "CONFIDENTIALITY", "FOR INTERNAL USE ONLY"
#     ]
#     pdf = fitz.open(stream=file_bytes, filetype="pdf")
#     output_text = []
#     for page in pdf:
#         blocks = page.get_text("dict")["blocks"]
#         for block in blocks:
#             if block['type'] == 0:
#                 for line in block["lines"]:
#                     for span in line["spans"]:
#                         text = span['text'].strip()
#                         font_size = span.get('size', 0)
#                         rotation = abs(span.get('rotation', 0))
#                         opacity = span.get('opacity', 1)
#                         color = span.get('color', None)

#                         if (5 < font_size < 20 and rotation < 10 and opacity > 0.9
#                                 and color not in [8421504, 12632256, 0xCCCCCC]
#                                 and not contains_watermark_keyword(text, watermark_keywords)):
#                             output_text.append(text)
#     pdf.close()
#     return "\n".join(output_text)


# def extract_pdf_title(content: bytes, fallback_filename: str) -> str:
#     def is_valid_title(text: str) -> bool:
#         text = text.strip()
#         if len(text) < 10:
#             return False
#         if text.lower() in {"untitled", "document", "new", "scan"}:
#             return False
#         if re.fullmatch(r"\d+", text):
#             return False
#         if not re.search(r"[a-zA-Z]", text):
#             return False
#         return True

#     def is_author_line(text: str) -> bool:
#         # Looks like list of names or affiliations
#         return bool(re.search(r"\b(?:[A-Z]\w+\s+[A-Z]\w+|\d)\b", text)) and len(text) < 100

#     def meaningful_word_count(text: str) -> int:
#         words = re.findall(r'\b\w+\b', text.lower())
#         return sum(1 for w in words if w not in ENGLISH_STOP_WORDS)

#     try:
#         doc = fitz.open(stream=BytesIO(content), filetype="pdf")

#         # 1. Try metadata
#         metadata_title = doc.metadata.get("title", "")
#         if is_valid_title(metadata_title):
#             return html.unescape(metadata_title.strip())

#         # 2. Visual font-based scanning
#         page = doc.load_page(0)
#         blocks = page.get_text("dict")["blocks"]
#         font_groups = defaultdict(list)

#         for block in blocks:
#             if block.get("type") != 0:
#                 continue
#             for line in block.get("lines", []):
#                 line_text = ""
#                 font_sizes = []
#                 for span in line.get("spans", []):
#                     txt = span.get("text", "").strip()
#                     if txt:
#                         line_text += txt + " "
#                         font_sizes.append(span.get("size", 0))
#                 avg_font = sum(font_sizes) / \
#                     len(font_sizes) if font_sizes else 0
#                 line_text = line_text.strip()
#                 if is_valid_title(line_text):
#                     font_groups[round(avg_font, 1)].append(line_text)

#         if not font_groups:
#             raise ValueError("No valid font-based lines found")

#         # Pick lines with largest font
#         largest_font = max(font_groups.keys())
#         candidates = font_groups[largest_font]

#         # Filter and prioritize
#         filtered = [
#             (meaningful_word_count(text), ":" in text, text)
#             for text in candidates
#             if not is_author_line(text)
#         ]
#         if filtered:
#             filtered.sort(reverse=True)
#             return html.unescape(filtered[0][2].strip())

#     except Exception as e:
#         print(f"[extract_pdf_title] Error: {e}")

#     return os.path.splitext(fallback_filename)[0]


# def sanitize_vector_id_title(title: str) -> str:
#     # Normalize Unicode to ASCII-compatible
#     ascii_title = unicodedata.normalize('NFKD', title).encode(
#         'ascii', 'ignore').decode('ascii')
#     # Replace non-alphanumerics with hyphens
#     return re.sub(r'[^a-zA-Z0-9_-]+', '-', ascii_title).strip('-').lower()


# def chunk_and_embed(text, book_title, filename):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
#     chunks = splitter.split_text(text)
#     vectors = embed_model.embed_documents(chunks)
#     safe_title = sanitize_vector_id_title(book_title)

#     return [
#         {
#             "id": f"{safe_title}-chunk-{i}",
#             "values": vector,
#             "metadata": {
#                 "chunk_text": chunk,
#                 "book_title": book_title,
#                 "filename": filename,  # ✅ Add this line
#                 "chunk_index": i,
#                 "keywords": extract_keywords(chunk),
#             }
#         }
#         for i, (chunk, vector) in enumerate(zip(chunks, vectors))
#     ]


# def store_pdf_in_pinecone(file_bytes: bytes, book_title: str, filename: str):
#     text = parse_pdf_file(file_bytes)
#     records = chunk_and_embed(text, book_title, filename)
#     index.upsert(vectors=records, namespace="example-namespace")


# def enhance_prompt(user_query: str) -> str:
#     query = user_query.strip()
#     query_lower = query.lower()

#     # Clean and standardize
#     query = query[0].upper() + query[1:] if query else ""

#     # Patterns that suggest already good prompts
#     good_starters = ("elaborate", "explain", "describe",
#                      "compare", "give", "provide", "what", "how", "why")
#     if query_lower.startswith(good_starters):
#         return query

#     # Keywords to detect intent
#     if len(query.split()) <= 3:
#         # If too short, likely a topic
#         return f"Explain the concept of {query} in detail with examples."

#     # If query contains verbs like "use", "impact", "role", etc.
#     if any(kw in query_lower for kw in ["impact", "importance", "role", "usage", "use", "application"]):
#         return f"Discuss the {query_lower} in depth with real-world examples."

#     # If it's a question without a question word
#     if query.endswith("?"):
#         return f"Answer the following question in detail: {query}"

#     # Generic fallback
#     return f"Explain in detail: {query}"


# def extract_query_keywords(user_query: str) -> List[str]:
#     keywords = extract_keywords(user_query)
#     return keywords


# # def retrieve_query_results(user_query: str):
# #     enhanced_query = enhance_prompt(user_query)
# #     print(enhanced_query)
# #     query_vector = embed_model.embed_query(enhanced_query)
# #     print("Query vector generated")
# #     keywords = extract_query_keywords(enhanced_query)
# #     print("Keywords extracted:", keywords)
# #     results = index.query(
# #         vector=query_vector,
# #         top_k=100,
# #         namespace="example-namespace",
# #         include_metadata=True,
# #         # optional keyword filtering
# #         filter={"keywords": {"$in": keywords}}
# #     )
# #     print("Query executed")
# #     print("Raw query results:", results)

# #     # return results['matches']
# #     if all('keywords' in match.get('metadata', {}) for match in results.get('matches', [])):
# #         results['matches'] = hybrid_rerank(results, keywords)
# #     return results['matches']

# def retrieve_query_results(user_query: str):
#     enhanced_query = enhance_prompt(user_query)

#     print("Enhanced query:", enhanced_query)

#     # 1. Generate query embedding
#     query_vector = embed_model.embed_query(enhanced_query)

#     print("Query vector generated")

#     # 2. Extract lexical keywords
#     keywords = extract_query_keywords(enhanced_query)

#     print("Keywords extracted:", keywords)

#     # 3. Retrieve candidates using dense retrieval ONLY
#     results = index.query(
#         vector=query_vector,
#         top_k=100,
#         namespace="example-namespace",
#         include_metadata=True
#     )

#     print("Dense retrieval completed")
#     print("Candidates retrieved:", len(results.get("matches", [])))

#     # 4. Hybrid reranking
#     matches = hybrid_rerank(
#         results.get("matches", []),
#         keywords,
#         alpha=0.7
#     )

#     print("Hybrid reranking completed")

#     return matches


# def retrieve_query_results_me(user_query: str, book_names: List[str]):
#     if not book_names:
#         print("No book names provided")
#         return []
#     enhanced_query = enhance_prompt(user_query)
#     print("Enhanced query:", enhanced_query)

#     query_vector = embed_model.embed_query(enhanced_query)
#     print("Query vector generated")

#     keywords = extract_query_keywords(enhanced_query)
#     print("Keywords extracted:", keywords)

#     # Build filter with book name constraint
#     filter_condition = {
#         "keywords": {"$in": keywords},
#         "filename": {"$in": book_names}
#     }

#     results = index.query(
#         vector=query_vector,
#         top_k=45,
#         namespace="example-namespace",
#         include_metadata=True,
#         filter=filter_condition
#     )

#     print("Query executed")
#     print("Query results:", results)

#     if all('keywords' in match.get('metadata', {}) for match in results.get('matches', [])):
#         results['matches'] = hybrid_rerank(results, keywords)

#     return results['matches']


# # def rerank_by_keyword_overlap(results, query_keywords):
# #     def score(match):
# #         chunk_keywords = match.get('metadata', {}).get('keywords', [])
# #         return len(set(chunk_keywords) & set(query_keywords))

# #     matches = results.get('matches', [])
# #     print("Reranking matches based on keyword overlap")
# #     if not matches:
# #         return []
# #     return sorted(matches, key=score, reverse=True)
# def calculate_keyword_score(match, query_keywords):
#     """
#     Calculates lexical overlap between query keywords
#     and chunk keywords.

#     Returns a normalized score between 0 and 1.
#     """
#     chunk_keywords = match.get("metadata", {}).get("keywords", [])

#     query_set = set(query_keywords)
#     chunk_set = set(chunk_keywords)

#     if not query_set or not chunk_set:
#         return 0.0

#     intersection = query_set & chunk_set

#     # Recall-style overlap:
#     # How much of the query's meaningful vocabulary
#     # appears in this chunk?
#     return len(intersection) / len(query_set)


# def hybrid_rerank(matches, query_keywords, alpha=0.7):
#     """
#     Combines semantic vector similarity with lexical keyword overlap.

#     final_score =
#         alpha * vector_similarity
#         + (1-alpha) * keyword_score

#     alpha=0.7 means:
#         70% semantic similarity
#         30% keyword relevance
#     """

#     if not matches:
#         return []

#     for match in matches:
#         vector_score = float(match.get("score", 0.0))

#         keyword_score = calculate_keyword_score(
#             match,
#             query_keywords
#         )

#         final_score = (
#             alpha * vector_score
#             + (1 - alpha) * keyword_score
#         )

#         match["vector_score"] = round(vector_score, 4)
#         match["keyword_score"] = round(keyword_score, 4)
#         match["hybrid_score"] = round(final_score, 4)

#     matches.sort(
#         key=lambda x: x["hybrid_score"],
#         reverse=True
#     )

#     return matches

# # ─── API Endpoints ───


# @app.post("/upload/")
# async def upload_files(files: List[UploadFile] = File(...)):
#     uploaded_titles = []
#     print("uploading")
#     for file in files:
#         content = await file.read()
#         # title = os.path.splitext(file.filename)[0]
#         title = extract_pdf_title(content, file.filename)
#         print(f"Processing file: {title}")
#         store_pdf_in_pinecone(content, title, file.filename)
#         uploaded_titles.append(title)
#     return {"message": "✅ Files processed successfully", "uploaded_titles": uploaded_titles}

# # @app.post("/query/", response_model=List[QueryResponse])


# @app.post("/query/")
# async def query_pdf(req: QueryRequest):
#     matches = retrieve_query_results(req.question)
#     print("Query started")
#     if not matches:
#         return JSONResponse(content={"message": "No data available", "results": []}, status_code=200)
#     print("Query will return results")
#     # Extract context for AI agent
#     # Group chunks by book
#     # Normalize scores
#     # max_score = max(match["score"] for match in matches) or 1e-6
#     # for match in matches:
#     #     match["norm_score"] = match["score"] / max_score
#     max_score = max(
#         match["hybrid_score"]
#         for match in matches
#     ) or 1e-6

#     for match in matches:
#         match["norm_score"] = (
#             match["hybrid_score"] / max_score
#         )

#     # Group by book
#     book_chunks = defaultdict(list)
#     book_scores = defaultdict(list)
#     print("Grouping chunks by book")
#     for match in matches:
#         book = match["metadata"].get("book_title", "Unknown")
#         chunk = match["metadata"].get("chunk_text", "")
#         if chunk:
#             book_chunks[book].append(chunk)
#             book_scores[book].append(match["norm_score"])
#     print(f"Found {len(book_chunks)} books with matching chunks")
#     # Add AI agent result
#     # responses = [
#     #     QueryResponse(
#     #         book=match["metadata"].get("book_title", "Unknown"),
#     #         score=match["score"],
#     #         text=clean_chunk(match["metadata"].get("chunk_text", ""))
#     #     )
#     #     for match in matches
#     # ]

#     # Add AI agent result at the beginning
#     # responses.insert(0, QueryResponse(
#     #     book="AI Agent",
#     #     score=1.0,  # you can keep it highest or just use -1 if not used
#     #     text=str(agent_output.raw)
#     # ))

#     # return responses

#     book_responses = []

#     for book, chunks in book_chunks.items():
#         print(f"Processing book: {book} with {len(chunks)} chunks")
#         agent_output = await generate_agent_response(req.question, chunks)
#         avg_score = sum(book_scores[book]) / len(book_scores[book])
#         book_responses.append({
#             "book": book,
#             # Optional — or compute average match score
#             "score": round(avg_score, 3),
#             "text": str(agent_output.raw)
#         })

#     return JSONResponse(content={"results": book_responses}, status_code=200)

#     # return {
#     #     # ✅ Only return agent’s summarized paragraphPDF Content Analyzer
#     #     "agent_response": str(agent_output.raw)
#     # }


# # @app.post("/query/", response_model=List[QueryResponse])
# # async def query_pdf(req: QueryRequest):
# #     matches = retrieve_query_results(req.question)
# #     print(" quesry started")

# #     if not matches:
# #         return JSONResponse(content={"message": "No data available", "results": []}, status_code=200)
# #     print(" quesry  will returnr started")
# #     return [
# #         QueryResponse(
# #             book=match["metadata"].get("book_title", "Unknown"),
# #             score=match["score"],
# #             text=match["metadata"].get("chunk_text", "")
# #         )
# #         for match in matches
# #     ]
# class QueryMeRequest(BaseModel):
#     question: str
#     book_names: List[str]  # List of book/pdf names to filter on


# @app.post("/queryme/")
# async def query_pdf_with_filter(req: QueryMeRequest):
#     print("QueryMe started")
#     if not req.book_names:
#         return JSONResponse(content={"message": "No book names provided", "results": []}, status_code=200)
#     all_matches = retrieve_query_results_me(req.question, req.book_names)

#     if not all_matches:
#         return JSONResponse(content={"message": "No data available", "results": []}, status_code=200)

#     # print(f"Filtering matches to specified books: {req.book_names}")
#     # # Filter matches based on the list of book names
#     # filtered_matches = [
#     #     match for match in all_matches
#     #     if match["metadata"].get("book_title", "").strip() in req.book_names
#     # ]

#     # Normalize scores
#     # max_score = max(match["score"] for match in all_matches) or 1e-6
#     # for match in all_matches:
#     #     match["norm_score"] = match["score"] / max_score
#     max_score = max(
#         match["hybrid_score"]
#         for match in all_matches
#     ) or 1e-6

#     for match in all_matches:
#         match["norm_score"] = (
#             match["hybrid_score"] / max_score
#         )

#     # Group by book
#     book_chunks = defaultdict(list)
#     book_scores = defaultdict(list)
#     print("Grouping filtered chunks by book")
#     for match in all_matches:
#         book = match["metadata"].get("book_title", "Unknown")
#         chunk = match["metadata"].get("chunk_text", "")
#         if chunk:
#             book_chunks[book].append(chunk)
#             book_scores[book].append(match["norm_score"])

#     print(f"Found {len(book_chunks)} books with matching chunks")

#     # Build responses using AI agent
#     book_responses = []
#     for book, chunks in book_chunks.items():
#         print(f"Processing book: {book} with {len(chunks)} chunks")
#         agent_output = await generate_agent_response(req.question, chunks)
#         avg_score = sum(book_scores[book]) / len(book_scores[book])
#         book_responses.append({
#             "book": book,
#             "score": round(avg_score, 3),
#             "text": str(agent_output)  # Ensure this is the agent's response
#         })

#     return JSONResponse(content={"results": book_responses}, status_code=200)


# @app.get("/")
# def hello():
#     return {"message": "Hello, this is the PDF Query API!"}


# @app.head("/")
# def head_root():
#     return Response(status_code=200)



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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from crewai import Agent, Task, Crew, LLM
from pinecone import Pinecone
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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
    model="gemini/gemini-1.5-flash",
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
        return (
            bool(
                re.search(
                    r"\b(?:[A-Z]\w+\s+[A-Z]\w+|\d)\b",
                    text
                )
            )
            and len(text) < 100
        )

    def meaningful_word_count(text: str) -> int:
        words = re.findall(r"\b\w+\b", text.lower())

        return sum(
            1
            for word in words
            if word not in ENGLISH_STOP_WORDS
        )

    try:
        doc = pymupdf.open(
            stream=BytesIO(content),
            filetype="pdf"
        )

        try:
            metadata_title = doc.metadata.get("title", "")

            if is_valid_title(metadata_title):
                return html.unescape(metadata_title.strip())

            page = doc.load_page(0)

            blocks = page.get_text("dict").get("blocks", [])
            candidates = []

            for block in blocks:
                if block.get("type") != 0:
                    continue

                for line in block.get("lines", []):
                    texts = []
                    font_sizes = []

                    for span in line.get("spans", []):
                        txt = span.get("text", "").strip()

                        if txt:
                            texts.append(txt)
                            font_sizes.append(span.get("size", 0))

                    if not texts:
                        continue

                    line_text = " ".join(texts).strip()

                    if not is_valid_title(line_text):
                        continue

                    avg_font = (
                        sum(font_sizes) / len(font_sizes)
                        if font_sizes
                        else 0
                    )

                    if is_author_line(line_text):
                        continue

                    candidates.append(
                        (
                            avg_font,
                            meaningful_word_count(line_text),
                            line_text
                        )
                    )

            if candidates:
                candidates.sort(
                    key=lambda x: (x[0], x[1]),
                    reverse=True
                )

                return html.unescape(
                    candidates[0][2].strip()
                )
        finally:
            doc.close()

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

    vectors = embed_model.embed_documents(chunks)

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

def hybrid_rerank(matches, query_keywords, alpha=0.7):
    if not matches:
        return []

    for match in matches:
        vector_score = float(
            match.get("score", 0.0)
        )

        keyword_score = calculate_keyword_score(
            match,
            query_keywords
        )

        hybrid_score = (
            alpha * vector_score
            + (1 - alpha) * keyword_score
        )

        match["vector_score"] = round(
            vector_score,
            4
        )

        match["keyword_score"] = round(
            keyword_score,
            4
        )

        match["hybrid_score"] = round(
            hybrid_score,
            4
        )

    matches.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return matches

# ============================================================
# RETRIEVE
# ============================================================

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

    matches = results.get("matches", [])

    print(
        "Dense matches:",
        len(matches)
    )

    matches = hybrid_rerank(
        matches,
        keywords,
        alpha=0.7
    )

    matches = matches[:20]

    print(
        "Final matches:",
        len(matches)
    )

    for match in matches[:5]:
        print(
            "MATCH:",
            match.get("score"),
            match.get("hybrid_score"),
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

    matches = results.get("matches", [])

    print(
        "Filtered dense matches:",
        len(matches)
    )

    matches = hybrid_rerank(
        matches,
        keywords,
        alpha=0.7
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
            match["hybrid_score"]
            for match in matches
        ) or 1e-6

        for match in matches:
            match["norm_score"] = (
                match["hybrid_score"]
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
            match["hybrid_score"]
            for match in matches
        ) or 1e-6

        for match in matches:
            match["norm_score"] = (
                match["hybrid_score"]
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
