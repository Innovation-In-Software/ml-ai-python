# Lab 2: Calling LLMs with AWS Bedrock

> **Machine Learning with AI and Python** · Day 3
> Prerequisite: Lab 1 (Day 3) complete

## The scenario

You know how LLMs see text. Now you will call one from Python. AWS Bedrock is a managed service that gives you API access to foundation models — Claude, Amazon Nova, Titan, and others — without managing infrastructure. One Python client, one API call, and you have a production-grade LLM in your code. This is the foundation every lab for the rest of the day is built on.

## Why this lab matters

Most real-world AI integrations are not pre-built products — they are Python code that calls an LLM API, shapes the conversation with a system prompt, maintains message history, and handles streaming output. Understanding these primitives means you can build anything: a chatbot, a data analyst, a document reviewer, an automated pipeline. You will also see how to measure token usage and swap models with one line, which are the two habits that keep AI integrations cheap and flexible.

## What you will do

- Configure a Bedrock client and make your first model call
- Shape model behaviour with a system prompt
- Maintain a multi-turn conversation with message history
- Stream responses token by token
- Read token usage and estimate cost
- Swap models and compare outputs on the same prompt

## Before you start

You will use Python with boto3. Create `bedrock_lab.py`.

```
pip install boto3 python-dotenv
```

**Create your Bedrock API key:**

1. Open the AWS Bedrock console and sign in
2. In the left navigation choose **Settings → API keys**
3. Choose **Create API key**, give it a name, and set the expiry to **5 days**
4. Copy the key — you cannot retrieve it again after closing the dialog

**Create a `.env` file** in the same folder as your script and paste your key in:

```
AWS_BEARER_TOKEN_BEDROCK=your-key-here
```

---

## Step 1: Create the Bedrock client and make your first call

**Why:** `boto3` is the AWS Python SDK. The `bedrock-runtime` client is what you use for model inference — sending prompts and receiving responses. The Converse API is the modern, model-agnostic way to call any Bedrock model: the same code works for Claude, Nova, Titan, and others.

```python
import boto3
import json
from dotenv import load_dotenv

load_dotenv()

client = boto3.client("bedrock-runtime", region_name="us-east-1")

MODEL = "us.amazon.nova-lite-v1:0"

response = client.converse(
    modelId=MODEL,
    messages=[
        {
            "role": "user",
            "content": [{"text": "What is machine learning in one sentence?"}]
        }
    ]
)

reply = response["output"]["message"]["content"][0]["text"]
print(reply)
```

**What to notice:** The messages format — a list of dicts with `role` and `content` — is the same structure you will use throughout the day. This mirrors the chat format you will see in every LLM API.

> [!NOTE]
> **Checkpoint:** You got a response from a live LLM. Note the response time — this is the baseline latency you are working with for the rest of the day.

## Step 2: System prompts — shape the model's behaviour

**Why:** A system prompt is a set of instructions given to the model before the conversation starts. It controls the model's persona, tone, scope, and output format. A well-written system prompt is often more impactful than any amount of fine-tuning — it is the primary tool for making a general model behave like a specialist.

```python
system_prompt = """You are a concise data science tutor for a professional training course.
- Answer in plain English, no jargon without explanation
- Keep responses to 3 sentences maximum
- If asked something outside data science, redirect politely"""

questions = [
    "What is overfitting?",
    "What is the best pizza topping?",
    "When should I use a random forest over logistic regression?",
]

for question in questions:
    response = client.converse(
        modelId=MODEL,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": question}]}]
    )
    print(f"Q: {question}")
    print(f"A: {response['output']['message']['content'][0]['text']}")
    print()
```

**What to notice:** The out-of-scope question should trigger the redirect. The system prompt is not visible to the user in a real application — it is the developer's layer of control over model behaviour.

> [!TIP]
> System prompts are the first place to look when a model behaves unexpectedly. Be explicit: "do not" is clearer than "avoid", and "respond in exactly N sentences" is clearer than "be brief".

---

## Step 3: Multi-turn conversation — maintaining history

**Why:** LLMs are stateless — they remember nothing between API calls. To have a conversation, you must send the entire history on every request. Maintaining that history list is your responsibility as the developer. This is the core loop inside every chatbot.

```python
def chat(client, model_id, system_prompt, history, user_input):
    history.append({
        "role": "user",
        "content": [{"text": user_input}]
    })

    response = client.converse(
        modelId=model_id,
        system=[{"text": system_prompt}],
        messages=history
    )

    reply = response["output"]["message"]["content"][0]["text"]
    history.append({
        "role": "assistant",
        "content": [{"text": reply}]
    })
    return reply, history


system = "You are a helpful ML tutor. Remember what the student tells you about themselves."
history = []

turns = [
    "Hi, I'm a data analyst who has been coding Python for 2 years.",
    "What topic should I focus on to move into ML engineering?",
    "How long would that realistically take someone with my background?",
]

for turn in turns:
    reply, history = chat(client, MODEL, system, history, turn)
    print(f"User: {turn}")
    print(f"Assistant: {reply}")
    print()

print(f"Total messages in history: {len(history)}")
```

**What to notice:** The third question ("someone with my background") works because the model has the full history and knows the user is a 2-year Python developer. Remove the history and it loses that context.

> [!WARNING]
> History grows with every turn. On a long conversation, you will eventually hit the context window limit. Production chatbots use summarisation or sliding windows to manage this — something to be aware of before building for scale.

## Step 4: Streaming responses

**Why:** By default, `converse` waits until the model finishes generating before returning anything. On longer responses, that wait is noticeable. Streaming delivers tokens as they are generated, which makes the application feel responsive. This is how every chat interface you have used actually works.

```python
response = client.converse_stream(
    modelId=MODEL,
    messages=[{
        "role": "user",
        "content": [{"text": "Explain what a transformer architecture is in 4 bullet points."}]
    }]
)

print("Streaming response:")
full_text = ""
for event in response["stream"]:
    if "contentBlockDelta" in event:
        chunk = event["contentBlockDelta"]["delta"].get("text", "")
        print(chunk, end="", flush=True)
        full_text += chunk

print("\n")
print(f"Total characters received: {len(full_text)}")
```

**What to notice:** Each `contentBlockDelta` event delivers a small piece of the response as it is generated. The `flush=True` forces the output to appear immediately rather than buffering. In a web app, you would push each chunk to the frontend via a streaming HTTP response.

---

## Step 5: Token usage and cost awareness

**Why:** Every API call returns the token counts for that request. Reading them tells you what you spent, helps you optimise prompts, and lets you build cost guardrails into applications. This is the habit that prevents surprise bills.

```python
prompts = [
    ("short",    "What is a neural network?"),
    ("detailed", "Explain backpropagation step by step, covering the chain rule, gradient flow through each layer, and how weight updates are calculated. Include an example with a simple two-layer network."),
    ("with context", "Given that we are building a fraud detection model with highly imbalanced data (0.1% positive rate), what approach would you recommend and why?"),
]

NOVA_LITE_INPUT_COST_PER_M  = 0.06
NOVA_LITE_OUTPUT_COST_PER_M = 0.24

print(f"{'prompt':<16} {'in tokens':>10} {'out tokens':>12} {'cost ($)':>12}")
print("-" * 54)

for label, prompt in prompts:
    resp = client.converse(
        modelId=MODEL,
        messages=[{"role": "user", "content": [{"text": prompt}]}]
    )
    usage = resp["usage"]
    cost = (usage["inputTokens"]  / 1_000_000 * NOVA_LITE_INPUT_COST_PER_M +
            usage["outputTokens"] / 1_000_000 * NOVA_LITE_OUTPUT_COST_PER_M)
    print(f"{label:<16} {usage['inputTokens']:>10} {usage['outputTokens']:>12} {cost:>12.6f}")
```

**What to notice:** Output tokens cost more than input tokens and are harder to control — the model decides how much to write. System prompts that enforce brevity ("respond in 3 sentences") directly reduce output token spend.

> [!NOTE]
> **Checkpoint:** Calculate how many calls like the "detailed" prompt you could make for $1. Is that more or less than you expected?

---

## Step 6: Swap models — Nova Micro vs Nova Lite vs Nova Pro

**Why:** One of the benefits of Bedrock's unified Converse API is that switching models is a one-line change. The Amazon Nova family spans from ultra-cheap to highly capable — comparing outputs on the same prompt is the fastest way to calibrate which model is worth paying for.

```python
MODEL_MICRO = "us.amazon.nova-micro-v1:0"
MODEL_LITE  = "us.amazon.nova-lite-v1:0"
MODEL_PRO   = "us.amazon.nova-pro-v1:0"

test_prompts = [
    "In one sentence: what is the difference between supervised and unsupervised learning?",
    "Write a Python function that calculates the Fibonacci sequence up to n terms.",
    "A customer churn model has 95% accuracy but only 20% recall on churned customers. Is this model useful? Why or why not?",
]

for prompt in test_prompts:
    print(f"Prompt: {prompt}")
    print()
    for name, model_id in [("Nova Micro", MODEL_MICRO), ("Nova Lite", MODEL_LITE), ("Nova Pro", MODEL_PRO)]:
        resp = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}]
        )
        reply = resp["output"]["message"]["content"][0]["text"]
        tokens = resp["usage"]["inputTokens"] + resp["usage"]["outputTokens"]
        print(f"  [{name} | {tokens} tokens]")
        print(f"  {reply[:200]}...")
        print()
    print("-" * 60)
```

**What to notice:** Simple factual questions look nearly identical across all three. Differences emerge on the nuanced reasoning prompt (the churn model question) — that is the signal for when to reach for a more capable model. Neither is universally better; the right choice depends on task complexity, latency, and cost.

> [!TIP]
> Experiment freely — swap `MODEL` at the top of your file to any of `MODEL_MICRO`, `MODEL_LITE`, or `MODEL_PRO` and rerun any earlier step. You will see different response styles, lengths, and reasoning depth on the exact same prompt. This is the fastest way to build intuition for model selection.

---

## On your own

1. Modify the system prompt in Step 2 to make the model respond only in bullet points and limit responses to five bullets maximum. Test it with three different questions and verify the constraint holds.
2. The history in Step 3 grows unbounded. Write a trimming function that keeps only the last N turns of conversation (plus the system prompt) and test that the model still gives coherent responses after trimming.
3. Run the cost comparison in Step 5 with `MODEL_MICRO` and `MODEL_PRO` instead of `MODEL_LITE`. Which model gives better value on the "with context" prompt — better output quality per dollar?

## Responsible AI

> [!IMPORTANT]
> - LLM outputs are non-deterministic and can be wrong. Never pipe model output directly into a consequential action (database write, email send, financial transaction) without a human review or a validation step.
> - System prompts are not a security boundary. A determined user can often override them with prompt injection. Do not rely on system prompts alone to enforce hard access controls.
> - Every token sent and received is logged by the cloud provider. Do not put real personal data, credentials, or sensitive business information into prompts during development.

## What you learned

- The Bedrock Converse API is a unified interface for calling any Bedrock model with the same code
- System prompts control model behaviour, persona, and scope before the conversation starts
- LLMs are stateless — conversation history is your responsibility and grows with every turn
- Streaming delivers tokens as generated, making applications feel responsive
- Token usage is measurable per call — reading it is how you control cost
- Swapping models is a one-line change; comparing outputs calibrates when to use which model

## Stretch goals

- Add a `max_tokens` parameter to `converse` (under `inferenceConfig`) and confirm it caps the output length. What happens to the response when it gets cut off mid-sentence?
- Build a simple cost tracker that accumulates token spend across multiple calls in a session and prints a running total after each call.
- Try setting `temperature` to `0.0` and `1.0` in `inferenceConfig` and run the same creative prompt both ways. How does the output differ, and when would you choose each setting?
