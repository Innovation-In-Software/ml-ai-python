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
    {
        "toolSpec": {
            "name": "get_price",
            "description": "Get the current price for a QuantumNet product.",
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
    {
        "toolSpec": {
            "name": "create_ticket",
            "description": "Create a support ticket for a product issue.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "product_id": {
                            "type": "string",
                            "description": "Product ID"
                        },
                        "issue": {
                            "type": "string",
                            "description": "Description of the issue"
                        }
                    },
                    "required": ["product_id", "issue"]
                }
            }
        }
    },
]


# ── Step 2: Tool executor ─────────────────────────────────────────────────────

def execute_tool(name, inputs):
    if name == "check_stock":
        item = products_table.get_item(Key={"product_id": inputs["product_id"]}).get("Item")
        if not item:
            return f"Product {inputs['product_id']} not found."
        return f"{item['name']} has {item['stock']} units in stock."

    if name == "get_price":
        item = products_table.get_item(Key={"product_id": inputs["product_id"]}).get("Item")
        if not item:
            return f"Product {inputs['product_id']} not found."
        return f"{item['name']} is priced at ${item['price']} {item['currency']}."

    if name == "create_ticket":
        ticket_id = str(uuid.uuid4())[:8]
        tickets_table.put_item(Item={
            "ticket_id":  ticket_id,
            "product_id": inputs["product_id"],
            "issue":      inputs["issue"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status":     "open"
        })
        return f"Support ticket created. Ticket ID: {ticket_id}."

    return f"Unknown tool: {name}"


# ── Step 3: Tool use loop ─────────────────────────────────────────────────────

def run_tools(question):
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

        if stop_reason == "tool_use":
            tool_results = []
            for block in response["output"]["message"]["content"]:
                if "toolUse" in block:
                    result = execute_tool(block["toolUse"]["name"], block["toolUse"]["input"])
                    tool_results.append({
                        "toolResult": {
                            "toolUseId": block["toolUse"]["toolUseId"],
                            "content":   [{"text": result}]
                        }
                    })
            messages.append({"role": "user", "content": tool_results})


# ── Step 4: Classifier ────────────────────────────────────────────────────────

def classify(question):
    response = bedrock.converse(
        modelId=MODEL_MICRO,
        messages=[{"role": "user", "content": [{"text": (
            "Classify this question as exactly 'docs' or 'tools'.\n"
            "docs: how to configure, set up, or troubleshoot a product.\n"
            "tools: stock levels, pricing, or creating a support ticket.\n"
            f"Question: {question}\n"
            "Respond with exactly one word."
        )}]}]
    )
    return response["output"]["message"]["content"][0]["text"].strip().lower()


# ── Step 5: Router ────────────────────────────────────────────────────────────

def answer(question, product_id=None):
    from rag import retrieve, generate
    category = classify(question)
    if category == "tools":
        return run_tools(question)
    chunks = retrieve(question, product_id)
    return generate(question, chunks)


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
