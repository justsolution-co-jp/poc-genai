from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

class RAGEngine:
    def __init__(self, doc_path="document.txt"):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = self._init_vectorstore(doc_path)

    def _init_vectorstore(self, doc_path):
        with open(doc_path, 'r', encoding='utf-8') as f:
            text = f.read()
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        docs = splitter.split_documents([Document(page_content=text)])
        return FAISS.from_documents(docs, self.embeddings)

    def retrieve(self, query, k=3):
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
