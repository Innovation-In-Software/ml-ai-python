"""
agent.py - Multi-model tool use agent
======================================
Two-path system:
  - Docs questions  -> RAG pipeline (rag.py from Lab 3)
  - Tools questions -> Bedrock tool use + DynamoDB

Complete each TODO in order. Test after each step.
"""
import boto3
import json
import os
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

REGION      = "us-east-1"
MODEL       = "us.amazon.nova-lite-v1:0"
MODEL_MICRO = "us.amazon.nova-micro-v1:0"

bedrock  = boto3.client("bedrock-runtime", region_name=REGION)
dynamodb = boto3.resource("dynamodb",      region_name=REGION)

products_table = dynamodb.Table("quantumnet-products")
tickets_table  = dynamodb.Table("quantumnet-tickets")


# ── Step 1: Tool definitions ──────────────────────────────────────────────────
# Each tool tells the model what it can call and what parameters to pass.
# The model never calls DynamoDB directly - it asks you to run the tool
# and sends the result back in the next message.

TOOLS = [
    {
        "toolSpec": {
            "name": "check_stock",
            "description": "Check the current stock level for a QuantumNet product.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "product_id": {
                            "type": "string",
                            "description": "Product ID, e.g. qr5000, qs24, qr500"
                        }
                    },
                    "required": ["product_id"]
                }
            }
        }
    },
    # TODO: add get_price tool definition
    #   - name: "get_price"
    #   - description: returns the price for a product
    #   - input: product_id (string, required)

    # TODO: add create_ticket tool definition
    #   - name: "create_ticket"
    #   - description: creates a support ticket and returns the ticket ID
    #   - inputs: product_id (string), issue (string) - both required
]


# ── Step 2: Tool executor ─────────────────────────────────────────────────────
# The model calls tools by name. This function receives the name and inputs
# and does the actual DynamoDB work.

def execute_tool(name, inputs):
    """Execute a tool call and return the result as a string."""

    if name == "check_stock":
        item = products_table.get_item(Key={"product_id": inputs["product_id"]}).get("Item")
        if not item:
            return f"Product {inputs['product_id']} not found."
        return f"{item['name']} has {item['stock']} units in stock."

    # TODO: handle "get_price"
    #   - get the item from products_table
    #   - return a string like "QuantumRouter QR-5000 is priced at $1299.99 USD."

    # TODO: handle "create_ticket"
    #   - generate a ticket_id with str(uuid.uuid4())[:8]
    #   - write to tickets_table: ticket_id, product_id, issue,
    #     created_at=datetime.now(timezone.utc).isoformat(), status="open"
    #   - return a string confirming the ticket ID

    return f"Unknown tool: {name}"


# ── Step 3: Tool use loop ─────────────────────────────────────────────────────
# The model may call multiple tools before giving a final answer.
# Keep looping until stopReason is "end_turn".

def run_tools(question):
    """Run the Bedrock tool use loop until the model returns a final answer."""

    messages = [{"role": "user", "content": [{"text": question}]}]

    while True:
        response = bedrock.converse(
            modelId=MODEL,
            messages=messages,
            toolConfig={"tools": TOOLS}
        )

        messages.append(response["output"]["message"])
        stop_reason = response["stopReason"]

        if stop_reason == "end_turn":
            return response["output"]["message"]["content"][0]["text"]

        # TODO: handle stop_reason == "tool_use"
        #   - loop over response["output"]["message"]["content"]
        #   - for each block that has a "toolUse" key:
        #       name        = block["toolUse"]["name"]
        #       inputs      = block["toolUse"]["input"]
        #       tool_use_id = block["toolUse"]["toolUseId"]
        #       result      = execute_tool(name, inputs)
        #   - build a list of toolResult content blocks and append as a "user" message:
        #       {"role": "user", "content": [
        #           {"toolResult": {"toolUseId": ..., "content": [{"text": result}]}}
        #       ]}
        #   - continue the loop


# ── Step 4: Classifier ────────────────────────────────────────────────────────
# Use Nova Micro to decide which path to take. Cheap and fast for a simple
# binary decision - save the heavier model for the actual work.

def classify(question):
    # TODO: call bedrock.converse() with MODEL_MICRO
    # Prompt the model to classify the question as exactly "docs" or "tools":
    #   - docs: how to configure, set up, or troubleshoot a product
    #   - tools: stock levels, pricing, or creating a support ticket
    # Return the response stripped and lowercased.
    pass


# ── Step 5: Router ────────────────────────────────────────────────────────────
# Connect the classifier to the two pipelines.
# This is the multi-model workflow: one model classifies, another answers.

def answer(question, product_id=None):
    # TODO:
    # 1. Call classify(question)
    # 2. If "tools" -> return run_tools(question)
    # 3. If "docs"  -> from rag import retrieve, generate
    #                  chunks = retrieve(question, product_id)
    #                  return generate(question, chunks)
    pass


# ── Manual test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("How do I configure a VLAN on the QR-5000?",   "qr5000"),
        ("How many QS-24 switches are in stock?",        None),
        ("What is the price of the QR-500?",             None),
        ("Create a ticket: QR-5000 keeps dropping BGP.", "qr5000"),
    ]

    for question, product_id in tests:
        print(f"Q: {question}")
        print(f"A: {answer(question, product_id)}")
        print()
