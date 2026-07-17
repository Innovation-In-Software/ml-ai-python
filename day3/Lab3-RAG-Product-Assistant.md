# Lab 3: RAG with Amazon Bedrock Knowledge Bases

> **Machine Learning with AI and Python** · Day 3
> Prerequisite: Lab 2 (Day 3) complete

## The scenario

QuantumNet makes enterprise networking hardware. Their support team answers hundreds of questions a week: "how do I configure a VLAN on the QR-5000?", "what is the default password for the QS-24?", "how does PoE priority work?" The answers are all in the product manuals. Your job is to build an AI assistant that retrieves the right section of the right manual and generates a grounded, accurate answer using AWS-managed infrastructure, the same way you would build it in production.

## Why this lab matters

In Lab 1 you built semantic search by hand using local embeddings. That taught you the mechanics. This lab shows you how it is done in production: documents stored in S3, a managed vector index in OpenSearch Serverless, and retrieval via Amazon Bedrock Knowledge Bases (the AWS equivalent of Azure AI Search). You write almost no infrastructure code. You focus on the retrieval quality, the generation prompt, and the product. That is the real job.

## What you will do

- Upload product manuals and metadata to S3
- Create and sync a Bedrock Knowledge Base in the AWS console
- Retrieve relevant chunks from the Knowledge Base with Python
- Generate grounded answers with Nova Lite
- Test the full pipeline in the browser via the Flask app

## Before you start

The starter app is in the `starter/` folder. Install dependencies:

```
cd starter
pip install -r requirements.txt
```

**Create your Bedrock API key** (if you have not already from Lab 2):

1. Open the AWS Bedrock console and sign in
2. In the left navigation choose **Settings → API keys**
3. Choose **Create API key**, give it a name, and set the expiry to **5 days**
4. Copy the key. You cannot retrieve it again after closing the dialog.

Copy `.env.example` to `.env` and fill in your credentials:

```
AWS_ACCESS_KEY_ID=your-access-key-here
AWS_SECRET_ACCESS_KEY=your-secret-key-here
AWS_BEARER_TOKEN_BEDROCK=your-bedrock-api-key-here
KB_ID=your-knowledge-base-id-here
```

You will fill in `KB_ID` after completing Step 2. The AWS access key and secret are needed for S3. The Bedrock API key covers all Bedrock calls.

You will need an S3 bucket. Create one in the AWS console (S3 → Create bucket → choose a unique name → region us-east-1 → all other defaults). Note the bucket name.

---

## Step 1: Upload the product manuals to S3

Bedrock Knowledge Bases pulls documents from S3. Each manual gets an accompanying metadata file that tags it with a `product_id`, which is what allows you to filter retrieval to a specific product later.

Run the setup script with your bucket name:

```
python setup_kb.py your-bucket-name
```

The script uploads three manuals and three matching `.metadata.json` files:

```
s3://your-bucket/manuals/qr5000.txt
s3://your-bucket/manuals/qr5000.txt.metadata.json   <- {"product_id": "qr5000"}
s3://your-bucket/manuals/qr500.txt
s3://your-bucket/manuals/qr500.txt.metadata.json    <- {"product_id": "qr500"}
s3://your-bucket/manuals/qs24.txt
s3://your-bucket/manuals/qs24.txt.metadata.json     <- {"product_id": "qs24"}
```

Verify in the S3 console that all six files appear before moving on.

> [!NOTE]
> **Checkpoint:** Six files in S3 - three `.txt` manuals and three `.metadata.json` files. If you only see three, re-run the script.

---

## Step 2: Create and sync the Bedrock Knowledge Base

The Knowledge Base handles everything a FAISS index did in Lab 1: chunking, embedding, and indexing into a managed vector store. You configure it once and AWS keeps the index in sync with your S3 bucket.

In the AWS console:

1. Navigate to **Amazon Bedrock → Knowledge Bases → Create Knowledge Base**
2. When prompted for the type, choose **Knowledge Base with vector store** (AWS managed, not "Custom" or "self-managed")
3. Name it `quantumnet-product-kb`
4. Under **Embeddings model**: leave the default **Managed embeddings model** selected - no additional cost, AWS manages the vector store for you
5. Under **Data source**: choose Amazon S3, then click **Browse S3** and select the `manuals/` folder in your bucket
6. Leave all other settings as defaults and click **Create Knowledge Base** - provisioning takes 1-2 minutes
7. Once created, click **Sync** to index the documents - wait for status to show **Available**
8. Copy the **Knowledge Base ID** from the overview page (format: `XXXXXXXXXX`)

Add the Knowledge Base ID to your `.env` file:

```
KB_ID=XXXXXXXXXX
```

> [!TIP]
> The managed vector store is a real OpenSearch Serverless collection running in AWS. Bedrock Knowledge Bases is a managed layer on top of it, the same technology you would use in a production RAG system.

## Step 3: Retrieve from the Knowledge Base

This step has no LLM. It is a pure vector database search. Your query gets converted to an embedding and matched against the indexed document chunks by similarity score. The `product_id` filter works like a SQL `WHERE` clause, scoping results to one product's manual. No generation happens here - you are confirming the right chunks come back before wiring up the model.

Open `rag.py` and implement `retrieve`:

```python
def retrieve(query, product_id=None, top_k=4):
    config = {"numberOfResults": top_k}

    if product_id:
        config["filter"] = {
            "equals": {"key": "product_id", "value": product_id}
        }

    response = BEDROCK_AGENT.retrieve(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": query},
        retrievalConfiguration={"managedSearchConfiguration": config}
    )

    return [
        {
            "text":   r["content"]["text"],
            "source": r["location"]["s3Location"]["uri"],
            "score":  r["score"]
        }
        for r in response["retrievalResults"]
    ]
```

Test it:

```python
# test_rag.py
from rag import retrieve

print("=== No filter ===")
for c in retrieve("default password"):
    print(f"  [{c['source'].split('/')[-1]}] score={c['score']:.3f}  {c['text'][:80]}...")

print("\n=== Filtered to qr5000 ===")
for c in retrieve("default password", product_id="qr5000"):
    print(f"  [{c['source'].split('/')[-1]}] score={c['score']:.3f}  {c['text'][:80]}...")
```

> [!NOTE]
> **Checkpoint:** Without the filter you see passwords from multiple products. With `product_id="qr5000"` you see only QR-5000 results. If scores are all below 0.3 or results are unrelated, check that the Knowledge Base sync completed.

## Step 4: Generate a grounded answer

Now that you have the right chunks, pass them to the model. The key instruction in the prompt is "answer ONLY from the provided context." This is what makes RAG reliable - the model reads real documentation rather than guessing from general knowledge. If the answer is not in the retrieved chunks, the model should say so rather than invent one.

Implement `generate` in `rag.py`:

```python
def generate(question, chunks):
    if not chunks:
        return "I could not find relevant information in the product manuals for that question."

    context = "\n\n---\n\n".join(
        f"[Source: {c['source'].split('/')[-1]}]\n{c['text']}"
        for c in chunks
    )

    prompt = f"""You are a product support assistant for QuantumNet networking equipment.
Answer the user's question using ONLY the information in the provided context.
If the answer is not in the context, say "I don't have that information in the product manuals."
Be concise and include exact values (IP addresses, passwords, menu paths) when present.

CONTEXT:
{context}

QUESTION: {question}"""

    response = BEDROCK.converse(
        modelId=MODEL,
        messages=[{"role": "user", "content": [{"text": prompt}]}]
    )
    return response["output"]["message"]["content"][0]["text"]
```

Test the full pipeline:

```python
# test_rag.py
from rag import retrieve, generate

question = "What is the default admin password for the QR-5000 and where do I log in?"
chunks   = retrieve(question, product_id="qr5000")
answer   = generate(question, chunks)
print(answer)
```

> [!NOTE]
> **Checkpoint:** The answer includes the exact default password and management URL from the manual, not a generic guess. Try asking about something not in any manual ("what is the warranty on the QA-300?") and verify the model correctly says it does not have that information.

---


## Step 5: Test in the browser

The Flask app in `app.py` already calls `rag.answer()`. Run it and verify the full end-to-end flow works through the UI.

```
python app.py
```

Open http://127.0.0.1:5000/chat and try these questions:

| Question | Expected behaviour |
|----------|-------------------|
| "How do I configure a VLAN trunk port on the QS-24?" | Steps from the QS-24 manual |
| "What is the default password for the QR-500?" | Exact value from qr500.txt |
| "What does error QR5-ERR-007 mean?" | BGP session explanation from qr5000.txt |
| "What is the best router for gaming?" | "I don't have that information" |

Use the product dropdown to filter. Answers should stay scoped to the selected product.

> [!NOTE]
> **Checkpoint:** The chat app answers product questions accurately from the manuals and declines to answer questions outside its knowledge base.

---

## On your own

1. Add a fourth product: write a short manual for a fictional `QuantumAP QA-300` access point (WiFi 6E, default IP 10.0.1.1, default password QAP@300admin, cover basic setup and channel configuration). Upload it with `setup_kb.py`, re-sync the Knowledge Base, add it to `products.json`, and verify the assistant answers questions about it without any code changes.
2. The current prompt instructs the model to answer "ONLY from context." Remove that instruction and ask "what is the default password for the QR-5000?" Does the model now mix in general knowledge? What is the risk of a model that does this in a real support tool?
3. Look at the relevance scores returned by `retrieve`. Set `top_k=8` and add a threshold: only pass chunks with `score > 0.5` to `generate`. Does answer quality improve or degrade? What does this tell you about retrieval quality on this dataset?

## Responsible AI

> [!IMPORTANT]
> - Grounding in retrieved documents reduces hallucination but does not eliminate it. Test with questions where you know the correct answer before exposing the system to real users.
> - Product manuals go out of date. When a product is updated, the Knowledge Base must be re-synced. Stale retrieval produces confidently wrong answers.
> - Do not upload documents containing personal data, credentials, or proprietary information to S3 without confirming your organisation's data handling and cloud storage policies.
> - Metadata filtering is a relevance tool, not a security boundary. Do not use it to hide documents from authorised users.

## What you learned

- Bedrock Knowledge Bases manages chunking, embedding, and indexing as a fully AWS-hosted service with no local vector database required
- Documents in S3 with accompanying `.metadata.json` files can be filtered at retrieval time, scoping answers to a specific product, category, or any custom attribute
- The `retrieve` + `generate` pattern gives full control over the generation prompt; `retrieve_and_generate` is faster to build with built-in citation support
- The system prompt instruction "answer ONLY from context" is what makes RAG trustworthy. Test that it works before shipping.
- Adding a new product requires no code changes. Upload documents, re-sync, and the Knowledge Base picks them up automatically.

## Stretch goals

- Modify `generate` to also return the list of source filenames alongside the answer, and display them in the chat UI as "Sources: qr5000.txt, qs24.txt" below each response.
- The Knowledge Base currently uses default chunking settings. In the AWS console, explore the **chunking strategy** options (Fixed size, Semantic, Hierarchical). Change to Semantic chunking, re-sync, and compare retrieval quality on a complex multi-step question.
- Implement streaming in the Flask endpoint: use `BEDROCK.converse_stream` in `generate` and return a `text/event-stream` HTTP response so the answer appears word by word in the browser chat UI.
