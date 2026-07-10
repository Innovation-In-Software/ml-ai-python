from flask import Flask, render_template, request, jsonify
from pathlib import Path
from dotenv import load_dotenv
import json
import rag

load_dotenv()

app = Flask(__name__)

with open("data/products.json") as f:
    PRODUCTS = json.load(f)

print("Bedrock Knowledge Base RAG pipeline ready.")


def get_product(product_id):
    return next((p for p in PRODUCTS if p["id"] == product_id), None)


@app.route("/")
def index():
    return render_template("index.html", products=PRODUCTS)


@app.route("/product/<product_id>")
def product(product_id):
    prod = get_product(product_id)
    if not prod:
        return "Product not found", 404
    return render_template("product.html", product=prod)


@app.route("/chat")
def chat_page():
    product_id = request.args.get("product_id")
    selected = get_product(product_id) if product_id else None
    return render_template("chat.html", products=PRODUCTS, selected=selected)


@app.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.get_json()
    question = (data or {}).get("question", "").strip()
    product_id = (data or {}).get("product_id")

    if not question:
        return jsonify({"error": "No question provided."}), 400

    answer = rag.answer(question, product_id)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
