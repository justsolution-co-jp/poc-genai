package com.example.java_rag.config;

import dev.langchain4j.model.ollama.OllamaChatModel;

public class OllamaConfig {
    public static OllamaChatModel localModel() {
        return OllamaChatModel.builder()
                .baseUrl("http://localhost:11434") // Ollama 服务地址
                .modelName("mistral") // 可换成 "codellama"、"mistral" 等
                .build();
    }
}
