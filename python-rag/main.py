from flask import Flask, request, jsonify
from rag_engine import RAGEngine

app = Flask(__name__)
engine = RAGEngine()

@app.route("/retrieval", methods=["POST"])
def retrieve():
    data = request.get_json()
    query = data.get("question")

    # 1. 相似文档检索
    chunks = engine.retrieve(query)

    # 2. 拼接 Prompt
    prompt = "\n\n".join(chunks) + f"\n\n用户问题：{query}"

    # 3. 返回给 Java 构造的内容
    return jsonify({
        "chunks": chunks,
        "prompt": prompt
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
