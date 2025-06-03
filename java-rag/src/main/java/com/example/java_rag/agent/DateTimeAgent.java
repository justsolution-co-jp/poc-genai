package com.example.java_rag.agent;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

public interface DateTimeAgent {

    @SystemMessage("""
            你是一个智能时间助手，可以获取
            """)
    @UserMessage("{input}")
    String chat(@V("input") String input);
}
