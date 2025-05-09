package com.example.java_rag.controller;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.example.java_rag.utils.PromptUtil;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

@RestController
@RequestMapping("/rag")
public class RagController {

    private final ObjectMapper mapper = new ObjectMapper();
    private final HttpClient client = HttpClient.newHttpClient();

    @PostMapping("/ask")
    public ResponseEntity<?> ask(@RequestBody Map<String, String> body) {
        String question = body.get("question");

        // 1. 调用 Python 分段服务
        List<String> chunks = getChunksFromPython(question);

        // 2. 构造 Prompt
        // String prompt = String.join("\n\n", chunks) + "\n\n用户问题：" + question;
        String prompt = PromptUtil.buildPromptWithLanguageHint(chunks, question);
        // 3. 调用 Ollama 本地模型
        String answer = callOllama(prompt);

        // 4. 返回结果
        return ResponseEntity.ok(Map.of(
                "chunks", chunks,
                "prompt", prompt,
                "answer", answer));
    }

    private List<String> getChunksFromPython(String question) {
        try {
            String json = mapper.writeValueAsString(Map.of("question", question));

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create("http://localhost:5002/retrieval"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(json))
                    .build();

            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            JsonNode chunksNode = mapper.readTree(response.body()).get("chunks");

            List<String> chunks = new ArrayList<>();
            for (JsonNode chunk : chunksNode) {
                chunks.add(chunk.asText());
            }

            return chunks;
        } catch (Exception e) {
            throw new RuntimeException("调用 Python 检索服务失败", e);
        }
    }

    private String callOllama(String prompt) {
        try {
            String json = mapper.writeValueAsString(Map.of(
                    "model", "mistral", // 可替换加载的模型名，如：phi、llama3 等
                    "prompt", prompt,
                    "stream", false));

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create("http://localhost:11434/api/generate"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(json))
                    .build();

            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            return mapper.readTree(response.body()).get("response").asText();
        } catch (Exception e) {
            throw new RuntimeException("调用 Ollama 模型失败", e);
        }
    }
}
