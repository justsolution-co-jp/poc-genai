package com.example.java_rag.service;

import java.util.List;

import org.springframework.stereotype.Service;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import jakarta.annotation.PostConstruct;

@Service
public class LangchainRagService {

    private EmbeddingStore<Document> embeddingStore;
    private EmbeddingModel embeddingModel;

    // @PostConstruct
    // public void init() {
    //     embeddingStore = FaissEmbeddingStore.builder()
    //             .withDimension(384)
    //             .withIndexPath("vector-index.index")
    //             .build();

    //     embeddingModel = new AllMiniLmL6V2EmbeddingModel();
    // }

    // public void indexDocuments(List<String> texts) {
    //     for (String text : texts) {
    //         Embedding embedding = embeddingModel.embed(text).content();
    //         embeddingStore.add(embedding, Document.from(text));
    //     }
    // }

    // public List<Document> searchRelevantChunks(String query, int topK) {
    //     Embedding queryEmbedding = embeddingModel.embed(query).content();
    //     return embeddingStore.findRelevant(queryEmbedding, topK);
    // }
}
