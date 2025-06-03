package com.example.java_rag.config;

public class OpenAiConfig {
    private static final String API_KEY;

    static {
        API_KEY = System.getenv("OPENAI_API_KEY");
        if (API_KEY == null || API_KEY.isBlank()) {
            throw new IllegalStateException("❌ 缺少环境变量 OPENAI_API_KEY");
        }
    }

    public static String getApiKey() {
        return API_KEY;
    }

    private OpenAiConfig() {
    }
}
