"""
rag.py — Product Manual RAG Pipeline (Bedrock Knowledge Bases)
===============================================================
Complete each TODO section in order. The app runs with placeholder responses
until you implement each step.

Prerequisites:
  - S3 bucket with manuals uploaded (run setup_kb.py)
  - Bedrock Knowledge Base created and synced (see Lab 3 Step 2)
  - KB_ID environment variable set, or replace the placeholder below
"""

import boto3
import os
from dotenv import load_dotenv

load_dotenv()

REGION       = "us-east-1"
MODEL = "us.amazon.nova-lite-v1:0"

BEDROCK       = boto3.client("bedrock-runtime",       region_name=REGION)
BEDROCK_AGENT = boto3.client("bedrock-agent-runtime", region_name=REGION)

# Set this to your Knowledge Base ID after completing Step 2
# You can also set the environment variable KB_ID instead
KNOWLEDGE_BASE_ID = os.environ.get("KB_ID", "YOUR_KB_ID_HERE")


# ── Step 3: Retrieve relevant chunks from the Knowledge Base ──────────────────

def retrieve(query, product_id=None, top_k=4):
    """
    Search the Bedrock Knowledge Base for chunks relevant to the query.
    If product_id is provided, filter results to that product only.

    Returns a list of dicts: {"text": str, "source": str, "score": float}

    Hint:
        Call BEDROCK_AGENT.retrieve() with:
          - knowledgeBaseId: KNOWLEDGE_BASE_ID
          - retrievalQuery: {"text": query}
          - retrievalConfiguration: {
                "managedSearchConfiguration": {
                    "numberOfResults": top_k,
                    # add "filter" here if product_id is set (see Step 4)
                }
            }

        Each result in response["retrievalResults"] has:
          - result["content"]["text"]          — the chunk text
          - result["location"]["s3Location"]["uri"] — the source file
          - result["score"]                    — relevance score
    """
    # TODO: implement
    return []  # placeholder


# ── Step 4: Add metadata filtering ───────────────────────────────────────────

def retrieve_filtered(query, product_id, top_k=4):
    """
    Same as retrieve() but scoped to a single product using metadata filtering.

    Hint: add a "filter" key to managedSearchConfiguration:
        "filter": {
            "equals": {"key": "product_id", "value": product_id}
        }
    """
    # TODO: implement (or merge into retrieve() with an if product_id: branch)
    return retrieve(query, top_k=top_k)  # placeholder — ignores product filter


# ── Step 5: Generate an answer with Claude ────────────────────────────────────

def generate(question, chunks):
    """
    Build a grounded prompt from the retrieved chunks and call Claude.
    Return the answer string.

    Hint:
        - Join chunks into a CONTEXT block.
        - Instruct Claude to answer ONLY from the context.
        - Use BEDROCK.converse() — same as Lab 2.
    """
    if not chunks:
        return "I could not find relevant information in the product manuals for that question."

    # TODO: implement
    return "RAG pipeline not yet fully implemented. Complete the TODO steps in rag.py."


# ── Public interface ──────────────────────────────────────────────────────────

def answer(question, product_id=None):
    """Full RAG pipeline: retrieve from Knowledge Base, then generate with Claude."""
    if KNOWLEDGE_BASE_ID == "YOUR_KB_ID_HERE":
        return "Knowledge Base ID not set. Add your KB ID to rag.py or set the KB_ID environment variable."
    chunks = retrieve(question, product_id)
    return generate(question, chunks)
