# Lab 1: How LLMs Read Text — Tokenization and Embeddings

> **Machine Learning with AI and Python** · Day 3
> Prerequisite: Day 2 complete

## The scenario

Before you call an LLM, it is worth understanding how one sees your text. An LLM does not read words or letters — it reads tokens, and it does not understand meaning through grammar rules — it understands meaning through vectors. These two ideas, tokenization and embeddings, are the foundation of everything you will build today: the Bedrock calls in Lab 2, the chat app in Lab 3, and especially the RAG pipeline in Lab 4, which is entirely built on embedding-based retrieval.

## Why this lab matters

When a prompt costs money, runs out of context, or returns a surprising result, the answer almost always comes back to tokens or embeddings. Understanding tokens tells you why prompts have limits and why wording matters. Understanding embeddings tells you how a model knows that "equity market decline" and "stocks fell sharply" mean the same thing without sharing a single word. By the end of this lab you will have built a semantic search engine — a miniature version of exactly what RAG does.

## What you will do

- Tokenize text and see exactly what an LLM reads
- Understand why token counts drive cost and context limits
- Encode sentences as embedding vectors
- Measure semantic similarity with cosine similarity
- Visualize a corpus in 2D embedding space
- Build a semantic search function over a small document set
- Understand how this connects to RAG

## Before you start

You will use Python with tiktoken, sentence-transformers, scikit-learn, and matplotlib. Create `embeddings_lab.py`.

```
pip install tiktoken sentence-transformers scikit-learn matplotlib
```

`sentence-transformers` will download a small model (~90 MB) on first run — this only happens once.

---

## Step 1: Tokenization — what an LLM actually reads

**Why:** LLMs do not process characters or words — they process tokens. A token is roughly a word or word-fragment. The tokenizer splits your text into tokens and converts each to an integer ID. This is the first transformation every prompt goes through before any intelligence happens.

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")   # encoding used by GPT-4 and Claude-family models

samples = [
    "The stock market crashed today.",
    "Investors panicked as share prices fell.",
    "ChatGPT is an LLM.",
    "Supercalifragilisticexpialidocious",
]

for text in samples:
    tokens = enc.encode(text)
    decoded = [enc.decode([t]) for t in tokens]
    print(f"text   : {text}")
    print(f"ids    : {tokens}")
    print(f"tokens : {decoded}")
    print(f"count  : {len(tokens)}")
    print()
```

**What to notice:** Common short words are usually one token. Long or rare words get split into multiple fragments. Punctuation is often its own token. The same sentence written differently can have a different token count.

> [!NOTE]
> **Checkpoint:** Find a word in your output that was split into more than two tokens. Why do you think that word got fragmented?

## Step 2: Token counts drive cost and context limits

**Why:** Every LLM API charges per token and enforces a context window limit (the maximum tokens the model can hold in memory at once). Counting tokens before sending a request tells you the cost and whether you will hit the limit.

```python
def count_tokens(text):
    return len(enc.encode(text))

texts = {
    "short prompt"        : "What is gradient descent?",
    "detailed prompt"     : "Explain gradient descent step by step, including the role of the learning rate, how it relates to the loss function, and what happens when the learning rate is too high or too low.",
    "typical paragraph"   : "Machine learning is a subfield of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.",
}

print(f"{'description':<22} {'tokens':>8}   {'rough cost @ $0.80/1M':>22}")
print("-" * 58)
for desc, text in texts.items():
    n = count_tokens(text)
    cost = n / 1_000_000 * 0.80
    print(f"{desc:<22} {n:>8}   ${cost:.8f}")
```

**What to notice:** Even a detailed paragraph is only a few hundred tokens. Context windows on modern models are in the hundreds of thousands of tokens. Cost per call at these rates is fractions of a cent — the economics only matter at scale.

> [!TIP]
> Claude 3.5 Haiku has a 200,000 token context window. At typical prompt lengths, you would need to send thousands of messages before the cost adds up to a dollar.

---

## Step 3: Embeddings — meaning as a vector

**Why:** Tokenization converts text to integers, but integers have no meaning relationship. Embeddings go further: they convert text into a dense vector of floating-point numbers (typically 384 or 1536 dimensions) where the position in that space encodes semantic meaning. Sentences that mean similar things land near each other in the space.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")   # fast, small, good quality

sentences = [
    "The Federal Reserve raised interest rates.",
    "Central bank increases borrowing costs.",
    "The cat sat on the mat.",
]

embeddings = model.encode(sentences)
print("embedding shape:", embeddings.shape)   # (3, 384) — 384 dimensions per sentence
print("first embedding (first 8 values):", embeddings[0][:8].round(3))
```

**What to notice:** Each sentence becomes a vector of 384 numbers. Those numbers encode meaning — the first two sentences should produce vectors that are close together even though they share almost no words.

## Step 4: Cosine similarity — how close is close?

**Why:** To compare two embeddings you measure the cosine of the angle between them. A score of 1.0 means identical meaning, 0.0 means unrelated, negative means opposing. This is the single operation at the heart of every semantic search and RAG retrieval.

```python
from sklearn.metrics.pairwise import cosine_similarity

pairs = [
    ("The Federal Reserve raised interest rates.",
     "Central bank increases borrowing costs."),
    ("The Federal Reserve raised interest rates.",
     "The cat sat on the mat."),
    ("Stock prices fell sharply.",
     "Equity markets declined today."),
]

print(f"{'similarity':>12}  pair")
print("-" * 80)
for a, b in pairs:
    emb_a = model.encode([a])
    emb_b = model.encode([b])
    score = cosine_similarity(emb_a, emb_b)[0][0]
    print(f"{score:>12.3f}  '{a[:40]}...' vs '{b[:40]}...'")
```

**What to notice:** Semantically similar sentences score high even with completely different words. Unrelated sentences score near zero. This is what makes embedding-based retrieval powerful — you are searching by meaning, not by keyword.

> [!NOTE]
> **Checkpoint:** You can explain why two sentences with no words in common can still have a high cosine similarity score.

---

## Step 5: Visualize the embedding space

**Why:** 384 dimensions are impossible to visualize directly. PCA compresses them to 2 so you can see the structure. Related sentences should cluster together on the plot, making the abstract concept of "meaning in space" concrete.

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

corpus = [
    "The stock market crashed today.",
    "Equity markets fell sharply on recession fears.",
    "Investors sold off shares amid panic.",
    "The Federal Reserve raised interest rates again.",
    "Central banks tightened monetary policy.",
    "Neural networks learn from labelled data.",
    "Deep learning models require GPUs to train.",
    "Transformers are the architecture behind LLMs.",
    "The weather is sunny and warm today.",
    "I enjoyed my walk in the park this morning.",
]

embeddings = model.encode(corpus)

pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

colors = ["steelblue"] * 3 + ["tomato"] * 2 + ["seagreen"] * 3 + ["gray"] * 2
labels = ["markets"] * 3 + ["central banks"] * 2 + ["ML/AI"] * 3 + ["unrelated"] * 2

plt.figure(figsize=(10, 7))
for i, (x, y) in enumerate(coords):
    plt.scatter(x, y, color=colors[i], s=80)
    plt.annotate(corpus[i][:35], (x, y),
                 textcoords="offset points", xytext=(6, 4), fontsize=8)
plt.title("Sentence embeddings in 2D (PCA) — related sentences cluster together")
plt.tight_layout()
plt.show()
```

**What to notice:** The market sentences, the central bank sentences, and the ML sentences should form rough clusters. The two unrelated sentences land away from the financial and technical groups.

## Step 6: Semantic search — find the most relevant documents

**Why:** This is the core operation of RAG. Given a query, embed it, then find the corpus documents with the highest cosine similarity to that query. No keywords, no exact matching — purely meaning-based retrieval.

```python
def semantic_search(query, corpus, corpus_embeddings, top_k=3):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, corpus_embeddings)[0]
    ranked = np.argsort(scores)[::-1][:top_k]
    return [(scores[i], corpus[i]) for i in ranked]

queries = [
    "What happened with interest rates?",
    "Tell me about AI and machine learning",
    "market selloff",
]

for query in queries:
    print(f"Query: {query}")
    results = semantic_search(query, corpus, embeddings)
    for score, doc in results:
        print(f"  [{score:.3f}] {doc}")
    print()
```

**What to notice:** The query "market selloff" retrieves the stock market sentences even though that exact phrase does not appear in the corpus. The model understands the meaning, not just the words.

> [!NOTE]
> **Checkpoint:** You can trace the full path from query string to retrieved result: encode → cosine similarity → rank → return top-k.

---

## Step 7: The connection to RAG

**Why:** What you just built in Step 6 is the retrieval half of Retrieval Augmented Generation. In Lab 4 you will do the same thing at a larger scale — embed real documents, store them in a vector database, and retrieve the most relevant ones at query time before passing them to the LLM as context.

```python
print("How semantic search becomes RAG:")
print()
print("1. INDEXING (done once, offline)")
print("   Load documents → chunk into paragraphs → embed each chunk → store in vector DB")
print()
print("2. RETRIEVAL (done at query time)")
print("   Embed the user's question → cosine similarity against stored chunks → top-k results")
print()
print("3. GENERATION (done at query time)")
print("   Inject retrieved chunks into the prompt → send to LLM → return grounded answer")
print()
print("You just built step 2. Lab 4 wires all three together.")
```

> [!TIP]
> The quality of a RAG system depends heavily on the embedding model and the chunking strategy, not just the LLM. A good retrieval step matters as much as a good generation step.

---

## On your own

1. Change the query in Step 6 to something completely unrelated to the corpus (for example, "best pizza recipes"). What similarity scores do you get, and what does that tell you about how to handle out-of-domain questions in a RAG system?
2. Add five more sentences to the corpus on a new topic of your choice and re-run the PCA plot. Do the new sentences form their own cluster?
3. The `all-MiniLM-L6-v2` model produces 384-dimensional embeddings. Look up `all-mpnet-base-v2` — it produces 768 dimensions. What trade-off are you making when you choose a larger embedding model?

## Responsible AI

> [!IMPORTANT]
> - Embedding models encode the biases present in their training data. Two sentences can land close in embedding space because they co-occur in biased text, not because they are genuinely similar in meaning.
> - Cosine similarity is a proxy for semantic relevance, not a guarantee of factual accuracy. A retrieved document may be topically related but factually wrong — the LLM in a RAG pipeline still needs to be prompted to verify what it uses.
> - Token counts determine what the model sees. If a prompt is too long, content gets cut off. Always know your token budget before designing a system that injects retrieved context.

## What you learned

- LLMs read tokens, not words — tokenization is the first step in every prompt
- Token counts drive cost and context limits, and you can measure them before sending a request
- Embeddings encode semantic meaning as vectors — similar meanings produce similar vectors
- Cosine similarity measures how close two meanings are in that vector space
- Semantic search retrieves documents by meaning, not keyword matching
- This retrieval operation is the foundation of RAG, which you will build in Lab 4

## Stretch goals

- Use `tiktoken` to count the tokens in a full page of text and estimate how many pages would fit in Claude's 200K context window.
- Try encoding the same sentence in English and its translation in another language — are the embeddings similar? What does that tell you about multilingual embeddings?
- Replace `cosine_similarity` with Euclidean distance (`np.linalg.norm(a - b)`) for the semantic search. Do the rankings change? Which metric is more robust and why?
