package com.example.java_rag.controller;

import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.example.java_rag.agent.CodeAgent;
import com.example.java_rag.utils.DateTimeTool;
import com.example.java_rag.utils.GitTool;
import com.example.java_rag.utils.PromptUtil;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.retriever.EmbeddingStoreRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;

@RestController
@RequestMapping("/rag")
public class RagController {

    private static final String OPENAI_API_KEY = System.getenv("OPENAI_API_KEY");

    private final ObjectMapper mapper = new ObjectMapper();
    private final HttpClient client = HttpClient.newHttpClient();

    // CodeAgent agent;

    // public RagController() {
    // // 初始化 LangChain4j Agent
    // this.agent = AiServices.builder(CodeAgent.class)
    // .chatLanguageModel(OllamaChatModel.builder()
    // .baseUrl("http://localhost:11434")
    // .modelName("mistral")
    // .timeout(Duration.ofSeconds(90))
    // .build())
    // .tools(new GitTool())
    // .build();
    // }

    private final CodeAgent agent;

    public RagController() {

        ChatLanguageModel openAiModel = OpenAiChatModel.builder()
                .apiKey(OPENAI_API_KEY)
                .modelName("gpt-4") // 或 gpt-3.5-turbo
                .timeout(Duration.ofSeconds(60))
                .build();

        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);
        this.agent = AiServices.builder(CodeAgent.class)
                .chatLanguageModel(openAiModel)
                .chatMemory(chatMemory)
                .tools(new DateTimeTool(), new GitTool()) // 示例工具
                .build();
    }

    @PostMapping("/ask")
    public ResponseEntity<?> ask(@RequestBody Map<String, String> body) {
        String question = body.get("question");
        System.out.println("question:" + question);

        // 1. 调用 Python 分段服务
        // List<String> chunks = getChunksFromPython(question);

        // 2. 构造 Prompt
        // String prompt = String.join("\n\n", chunks) + "\n\n用户问题：" + question;
        // String prompt = PromptUtil.buildPromptWithLanguageHint(Arrays.asList(),
        // question);
        // // 3. 调用 Ollama 本地模型
        // String answer = callChatGPT(prompt);
        String answer = agent.chat(question);
        // String answer = "今天是星期二，日期是2025年5月28日。";

        // String audioPath = callTTS(answer, PromptUtil.getTTSLanguage(question));
        // 4. 返回结果
        return ResponseEntity.ok(Map.of(
                // "chunks", chunks,
                // "prompt", prompt,
                // "audio", audioPath,
                "answer", answer));
    }

    @PostMapping("/agent")
    public ResponseEntity<String> agent(@RequestBody String command) {
        System.out.println("接收到指令>>>>" + command);
        String result = agent.chat(command);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/audio")
    public ResponseEntity<Resource> getAudio(@RequestParam String path) throws IOException {
        Path audioPath = Paths.get(path);
        if (!Files.exists(audioPath)) {
            return ResponseEntity.notFound().build();
        }

        Resource resource = new UrlResource(audioPath.toUri());

        return ResponseEntity.ok()
                .contentType(MediaType.parseMediaType("audio/wav"))
                .header(HttpHeaders.CONTENT_DISPOSITION, "inline; filename=" + audioPath.getFileName())
                .body(resource);
    }

    @PostMapping("/generate")
    public ResponseEntity<?> generateAudio(@RequestBody Map<String, String> body) {
        String text = body.get("text");
        String language = body.get("language_id");

        // 参数校验
        if (text == null || text.trim().isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("error", "参数 text 不能为空"));
        }
        if (language == null || language.trim().isEmpty()) {
            language = "zh"; // 默认中文
        }

        try {
            // 生成音频
            String path = callTTS(text, language);
            if (path == null) {
                return ResponseEntity.status(500).body(Map.of("error", "TTS 合成失败"));
            }

            return ResponseEntity.ok(Map.of(
                    "path", path,
                    "url", "/audio?path=" + URLEncoder.encode(path, StandardCharsets.UTF_8)));
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(500).body(Map.of("error", "TTS 请求出错", "detail", e.getMessage()));
        }
    }

    private List<String> getChunksFromPython(String question) {
        try {
            String json = mapper.writeValueAsString(Map.of("question", question));

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create("http://localhost:8888/retrieval"))
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
                    "model", "phi", // 可替换加载的模型名，如：phi、mistral 等
                    "prompt", prompt,
                    "stream", false));

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create("http://localhost:11434/api/generate"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(json))
                    .build();

            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            System.out.println("💬 Ollama 返回原始内容：" + response.body());
            return mapper.readTree(response.body()).get("response").asText();
        } catch (Exception e) {
            throw new RuntimeException("调用 Ollama 模型失败", e);
        }
    }

    public String callChatGPT(String prompt) {
        try {
            // 构造请求 JSON
            String json = mapper.writeValueAsString(Map.of(
                    "model", "gpt-4", // 或 gpt-3.5-turbo
                    "messages", List.of(Map.of(
                            "role", "user",
                            "content", prompt)),
                    "temperature", 0.7));

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create("https://api.openai.com/v1/chat/completions"))
                    .header("Authorization", "Bearer " + OPENAI_API_KEY)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(json))
                    .build();

            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

            System.out.println("💬 ChatGPT 返回原始内容：" + response.body());

            return mapper.readTree(response.body())
                    .get("choices").get(0).get("message").get("content").asText();

        } catch (Exception e) {
            throw new RuntimeException("调用 ChatGPT 失败", e);
        }
    }

    public String callTTS(String text, String languageId) {
        try {
            HttpClient client = HttpClient.newHttpClient();
            // 谁的声音
            // String speakerWavPath = "/path/to/your/speaker.wav";

            // 构建 URL 参数字符串
            // String textParam = URLEncoder.encode("你好，这是测试语音。", StandardCharsets.UTF_8);
            // String speaker = URLEncoder.encode("Daisy Studious", StandardCharsets.UTF_8);

            String encodedText = URLEncoder.encode(text, StandardCharsets.UTF_8);

            String url = String.format(
                    "http://localhost:5002/api/tts?text=%s",
                    encodedText);

            // 构建 GET 请求
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .GET()
                    .build();

            HttpResponse<byte[]> response = client.send(request, HttpResponse.BodyHandlers.ofByteArray());

            // 将音频保存为文件
            String outputPath = "/tmp/tts_output.wav";
            Files.write(Paths.get(outputPath), response.body());

            return outputPath;
        } catch (Exception e) {
            System.out.println(e);
            e.printStackTrace();
            return null;
        }
    }

    public static void main(String[] args) {
        RagController controller = new RagController();
        controller.callTTS("こんにちは", "ja");
    }

}
