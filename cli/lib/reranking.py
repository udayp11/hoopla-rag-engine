import os
from time import sleep
import json

from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"
cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")


def llm_rerank_individual(
    query: str, documents: list[dict], limit: int = 5
) -> list[dict]:
    scored_docs = []

    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""

        response = client.models.generate_content(model=model, contents=prompt)
        score_text = (response.text or "").strip()
        score = int(score_text)
        scored_docs.append({**doc, "rerank_score": score})

    scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored_docs[:limit]

def llm_rerank_batch(query: str, documents: list[dict], limit: int = 5) -> list[dict]:
    if not documents:
        return []

    doc_map = {}
    doc_list = []
    for i, doc in enumerate(documents[:20], 1):
        doc_id = f"doc_{i}"
        doc_map[doc_id] = doc
        doc_list.append(
            f"{doc_id}: {doc.get('title', '')} - {doc.get('document', '')[:200]}"
        )

    doc_list_str = "\n".join(doc_list)

    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first).
Format: doc_1, doc_2, doc_3, ...

Ranking:"""


    response = client.models.generate_content(model=model, contents=prompt)
    ranking_text = (response.text or "").strip()

     # Parse the ranking
    ranked_ids = []
    for part in ranking_text.replace("[", "").replace("]", "").split(","):
        doc_id = part.strip()
        if doc_id in doc_map:
            ranked_ids.append(doc_id)

    # Build reranked list
    reranked = []
    for doc_id in ranked_ids:
        if doc_id in doc_map:
            reranked.append(doc_map[doc_id])

    # Add any missing documents at the end
    for doc_id, doc in doc_map.items():
        if doc_id not in ranked_ids:
            reranked.append(doc)

    return reranked[:limit]

def cross_encoder_rerank(
    query: str, documents: list[dict], limit: int = 5
) -> list[dict]:
    pairs = []
    for doc in documents:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])

    scores = cross_encoder.predict(pairs)

    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)

    documents.sort(key=lambda x: x["rerank_score"], reverse=True)
    return documents[:limit]



def rerank(
    query: str, documents: list[dict], method: str = "batch", limit: int = 5
) -> list[dict]:
    if method == "individual":
        return llm_rerank_individual(query, documents, limit)
    if method == "batch":
        return llm_rerank_batch(query, documents, limit)
    if method == "cross_encoder":
        return cross_encoder_rerank(query, documents, limit)
    else:
        return documents[:limit]
