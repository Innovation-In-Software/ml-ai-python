# Lab 4: Multi-Model Workflow with Bedrock Tool Use

> **Machine Learning with AI and Python** · Day 3
> Prerequisite: Lab 3 (Day 3) complete

## The scenario

QuantumNet's support assistant handles two types of questions. Documentation questions ("how do I configure a VLAN?") are answered by the RAG pipeline from Lab 3. Business queries ("is the QR-5000 in stock?", "create a support ticket") need live data from a database. A single model doing everything is both wasteful and inflexible. This lab wires up a two-path system: a lightweight classifier routes each question to the right model and the right data source.

## What you will do

- Seed product inventory and ticket data into DynamoDB
- Define tools that let the model query and write that data
- Implement the tool use loop that drives the model until it has an answer
- Build a classifier that decides which path each question takes
- Connect the classifier to the RAG pipeline and the tool use pipeline

## Before you start

You need DynamoDB permissions. Your existing AWS credentials from Lab 3 cover this. Run the setup script to create the tables and seed the data:

```
python setup_dynamo.py
```

You should see three products created: qr5000, qr500, qs24.

All your work goes in `agent.py`. The file is already structured with the five steps as TODOs.

---

## Step 1: Add the missing tool definitions

Open `agent.py`. The `TOOLS` list already has `check_stock` defined. Study its structure, then add the two missing tools below it.

`get_price` — takes a `product_id`, returns the price for that product.

`create_ticket` — takes a `product_id` and an `issue` string, creates a support ticket and returns the ticket ID.

The tool definition tells the model what the tool does and what parameters to pass. The model never touches DynamoDB directly - it asks you to run the tool by name.

Test by printing `TOOLS` and confirming all three definitions look right before moving on.

---

## Step 2: Implement the tool executor

`execute_tool(name, inputs)` receives a tool name and the inputs the model chose. `check_stock` is already implemented. Add the other two:

For `get_price`: query `products_table` by `product_id` and return a string with the product name and price.

For `create_ticket`: generate a short ticket ID with `str(uuid.uuid4())[:8]`, write a record to `tickets_table` with `product_id`, `issue`, `created_at`, and `status="open"`, then return a confirmation string with the ticket ID.

Test it directly:

```python
print(execute_tool("check_stock", {"product_id": "qr5000"}))
print(execute_tool("get_price",   {"product_id": "qs24"}))
print(execute_tool("create_ticket", {"product_id": "qr500", "issue": "unit not booting"}))
```

Verify the ticket appears in the DynamoDB console before moving on.

---

## Step 3: Complete the tool use loop

`run_tools(question)` already calls the model and handles `end_turn`. Your job is to handle `tool_use`.

When `stopReason` is `"tool_use"`, the model's response contains one or more `toolUse` content blocks. For each one:

1. Extract `name`, `input`, and `toolUseId` from `block["toolUse"]`
2. Call `execute_tool(name, input)` to get the result
3. Build a `toolResult` content block with the `toolUseId` and result text
4. Append all results as a new `"user"` message and loop back

The model will keep calling tools until it has enough information to return a final answer. The loop exits when `stopReason` is `"end_turn"`.

Test it:

```python
print(run_tools("How many QR-5000 routers are in stock?"))
print(run_tools("What does the QS-24 cost?"))
print(run_tools("Create a support ticket for the QR-500: unit keeps rebooting."))
```

---

## Step 4: Build the classifier

`classify(question)` uses Nova Micro to decide whether a question needs `"docs"` or `"tools"`. Nova Micro is cheap and fast. It is the right model for a simple binary decision.

Write a prompt that:
- Explains the two categories (docs: configuration and troubleshooting; tools: stock, pricing, tickets)
- Asks the model to respond with exactly one word

Return the response stripped and lowercased. It should always be `"docs"` or `"tools"`.

Test it on several questions and confirm the classification is correct before wiring it into the router.

---

## Step 5: Connect the router

`answer(question, product_id)` is where everything comes together. Implement it:

1. Call `classify(question)`
2. If the result is `"tools"`, call `run_tools(question)` and return the answer
3. If the result is `"docs"`, import `retrieve` and `generate` from `rag`, retrieve chunks filtered by `product_id`, and return the generated answer

Run the full test suite at the bottom of `agent.py`:

```
python agent.py
```

All four test questions should route correctly and return meaningful answers.

> [!NOTE]
> **Checkpoint:** The VLAN question returns a configuration answer from the manual. The stock, price, and ticket questions return live data from DynamoDB. If a docs question returns a tools answer or vice versa, revisit your classifier prompt.

---

## On your own

1. Wire `agent.answer()` into the Flask app. Replace the call to `rag.answer()` in `app.py` with `agent.answer()` and confirm the chat UI handles both question types correctly.

2. Add a fourth tool: `list_open_tickets(product_id)` that queries `quantumnet-tickets` for all open tickets for a given product. Add the tool definition, implement it in `execute_tool`, and test it by creating a few tickets and then listing them.

3. The classifier sometimes misfires on ambiguous questions like "what is the QR-5000?". Add a third category `"both"` and handle it in the router by running both pipelines and combining the answers.

## Responsible AI

> [!IMPORTANT]
> - The model chooses which tools to call and what inputs to pass. Always validate inputs in `execute_tool` before writing to a database. Never trust the model's output as a safe database key.
> - Tool use loops can run indefinitely if the model keeps calling tools. Add a max iterations guard in production to prevent runaway API spend.
> - `create_ticket` writes real records on every call. During development, check the DynamoDB console regularly and delete test records so they do not pollute production data.

## What you learned

- Bedrock tool use lets the model call functions by name and incorporates results before answering
- The tool use loop runs until the model signals `end_turn`, which may require multiple tool calls
- A lightweight classifier model (Nova Micro) routing to specialist pipelines is cheaper and more maintainable than one model doing everything
- RAG and tool use solve different problems: RAG answers questions from static documents, tool use answers questions that require live or writable data

## Stretch goals

- The classifier uses a single prompt with no examples. Add three few-shot examples per category and measure whether ambiguous questions classify more reliably.
- Replace the binary classifier with a structured output approach: ask the model to return JSON with `{"category": "docs"|"tools", "confidence": 0.0-1.0}` and only route to tools if confidence is above 0.8, otherwise defaulting to RAG.
- Add response streaming to `run_tools` using `converse_stream` so the final answer appears token by token in the terminal.
