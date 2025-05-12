
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant

import os

class RAGEngine:
    def __init__(self, doc_path="document.txt"):
        print("🔧 初始化 RAG 引擎...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.collection_name = "my_docs"
        self.qdrant_client = QdrantClient(host="localhost", port=6333)

        # 初始化向量库，如果 collection 已存在则跳过导入
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
            embedding_function=self.embeddings,  # ✅ 正确参数名
            client=self.qdrant_client,
            collection_name=self.collection_name
        )

    def retrieve(self, query, k=3):
        print(f"🔍 正在检索与查询 '{query}' 最相似的 {k} 条内容...")
        docs = self.vectorstore.similarity_search(query, k=k)
        print("✅ 检索完成")
        return [doc.page_content for doc in docs]
