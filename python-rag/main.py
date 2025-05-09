from flask import Flask, request, jsonify
from rag_engine import RAGEngine

app = Flask(__name__)
engine = RAGEngine()

@app.route("/retrieval", methods=["POST"])
def retrieve():
    query = request.json.get("question")
    chunks = engine.retrieve(query)
    return jsonify({"chunks": chunks})

if __name__ == "__main__":
    app.run(port=5002)
