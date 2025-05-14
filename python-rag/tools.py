from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain.tools import tool
from pydantic import BaseModel, Field

import os
import subprocess
import re

# 获取 Git 仓库根目录路径
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class RAGEngine:
    def __init__(self, doc_path="document.txt"):
        print("🔧 初始化 RAG 引擎...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.collection_name = "my_docs"
        self.qdrant_client = QdrantClient(host="localhost", port=6333)

        if not self.qdrant_client.collection_exists(self.collection_name):
            print(f"📄 未检测到 collection '{self.collection_name}'，正在从 {doc_path} 初始化...")
            self._init_vectorstore(doc_path)
            print("✅ 向量数据库初始化完成")
        else:
            print(f"✅ collection '{self.collection_name}' 已存在，跳过初始化")

        self.vectorstore = Qdrant(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )
        print(f"🧩 当前 vectorstore 对象: {self.vectorstore}")

    def _init_vectorstore(self, doc_path):
        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"❌ 文档文件不存在: {doc_path}")

        with open(doc_path, 'r', encoding='utf-8') as f:
            text = f.read()

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        documents = splitter.split_documents([Document(page_content=text)])

        print(f"📦 共生成 {len(documents)} 个文档片段，开始写入 Qdrant...")

        Qdrant.from_documents(
            documents,
            embedding_function=self.embeddings,
            client=self.qdrant_client,
            collection_name=self.collection_name
        )

    def retrieve(self, query, k=3):
        print(f"🔍 正在检索与查询 '{query}' 最相似的 {k} 条内容...")
        docs = self.vectorstore.similarity_search(query, k=k)
        print("✅ 检索完成")
        return [doc.page_content for doc in docs]


class GitCommitParams(BaseModel):
    message: str = Field(..., description="提交信息，例如：更新功能模块")
    path: str = Field(".", description="提交的文件或目录路径，默认是 .")
    hint: str = Field("", description="额外的提示文字，例如 '推送'，表示执行 git push")


@tool("git_commit_and_optional_push")
def git_commit_and_optional_push(raw_input: str) -> str:
    """
    临时版本：用于测试 git commit 和 push。
    输入格式为 JSON 字符串，例如：
    {"message": "更新 main.py", "path": "python-rag/main.py", "hint": "推送"}
    """
    import json

    print("🚀 工具函数触发 ✅")
    print(f"raw_input: {raw_input}")

    try:
        params = json.loads(raw_input)
        message = params.get("message", "")
        path = params.get("path", ".")
        hint = params.get("hint", "")
    except Exception as e:
        return f"❌ 参数解析失败: {str(e)}"

    try:
        subprocess.run(["git", "add", path], check=True)
        subprocess.run(["git", "commit", "-m", message], check=True)
        result = f"✅ 已成功提交 `{path}`，信息: {message}\n"

        # 总是执行 push（为了测试）
        current_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            text=True
        ).stdout.strip()

        subprocess.run(["git", "push", "origin", current_branch], check=True)
        result += f"✅ 已成功推送到远程分支 `{current_branch}`"

        return result
    except subprocess.CalledProcessError as e:
        return f"❌ Git 操作失败: {str(e)}"


if __name__ == "__main__":
    print("这是 RAG 引擎模块")
