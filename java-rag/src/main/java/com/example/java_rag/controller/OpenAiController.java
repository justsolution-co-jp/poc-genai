package com.example.java_rag.controller;

import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.example.java_rag.config.OpenAiConfig;
import com.example.java_rag.utils.PromptUtil;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.core.JsonObject;
import com.openai.core.JsonValue;
import com.openai.models.ChatModel;
import com.openai.models.FunctionDefinition;
import com.openai.models.FunctionParameters;
import com.openai.models.chat.completions.ChatCompletion;
import com.openai.models.chat.completions.ChatCompletionAssistantMessageParam;
import com.openai.models.chat.completions.ChatCompletionAssistantMessageParam.Content;
import com.openai.models.chat.completions.ChatCompletionCreateParams;
import com.openai.models.chat.completions.ChatCompletionFunctionCallOption;
import com.openai.models.chat.completions.ChatCompletionMessage;
import com.openai.models.chat.completions.ChatCompletionMessageToolCall;
import com.openai.models.chat.completions.ChatCompletionTool;
import com.openai.models.chat.completions.ChatCompletionToolMessageParam;
import com.openai.models.models.Model;
import com.openai.models.responses.FunctionTool;
import com.openai.services.blocking.ModelService;

import static com.openai.core.ObjectMappers.jsonMapper;

@RestController
@RequestMapping("/openAi")
public class OpenAiController {

    private OpenAIClient openAIClient;

    public OpenAiController() {
        openAIClient = OpenAIOkHttpClient
                .builder()
                .apiKey(OpenAiConfig.getApiKey())
                .build();

    }

    @RequestMapping("modelList")
    public ResponseEntity<?> modelList() {
        List<String> modelList = openAIClient.models().list().items().stream().map(Model::id).toList();

        return ResponseEntity.ok(modelList);
    }

    @PostMapping("chat")
    public ResponseEntity<?> chat(@RequestBody Map<String, String> body) {
        String question = body.get("question");
        System.out.println("question:" + question);

        ChatCompletionCreateParams chatParams = ChatCompletionCreateParams
                .builder()
                .model(ChatModel.GPT_4O)
                .addUserMessage(question)
                .n(1) // > 1 时 多个回答 choices
                .build();
        ChatCompletion chatCompletion = openAIClient.chat().completions().create(chatParams);
        String content = chatCompletion.choices().get(0).message().content().get();
        return ResponseEntity.ok(Map.of(
                "answer", content));
    }

    @PostMapping("chatWithAudio")
    public ResponseEntity<?> chatWithAudio(@RequestBody Map<String, String> body) {
        String question = body.get("question");
        System.out.println("question:" + question);

        ChatCompletionCreateParams chatParams = ChatCompletionCreateParams
                .builder()
                .model(ChatModel.GPT_4O)
                .addUserMessage(question)
                .n(1) // > 1 时 多个回答 choices
                .build();
        ChatCompletion chatCompletion = openAIClient.chat().completions().create(chatParams);
        Optional<String> content = chatCompletion.choices().get(0).message().content();

        // String audioPath = callTTS(answer, PromptUtil.getTTSLanguage(question));

        return ResponseEntity.ok(Map.of(
                "answer", content.isPresent() ? content.get() : ""));
    }

    @PostMapping("chatWithAgent")
    public ResponseEntity<?> chatWithAgent(@RequestBody Map<String, String> body) {
        String question = body.get("question");
        System.out.println("🟡 提问：" + question);

        if (containsWakeWord(question)) {
            return ResponseEntity.ok(Map.of("answer", "我在，请问有什么需要帮助？"));
        }

        // 1. 构造 Tool
        List<ChatCompletionTool> tools = List.of(
                ChatCompletionTool.builder()
                        .function(FunctionDefinition.builder()
                                .name("get-current-time")
                                .description("返回当前东京时间")
                                .parameters(FunctionParameters.builder()
                                        .putAdditionalProperty("type", JsonValue.from("object"))
                                        .putAdditionalProperty("properties", JsonValue.from(Collections.emptyMap()))
                                        .putAdditionalProperty("required", JsonValue.from(Collections.emptyList()))
                                        .build())
                                .build())
                        .build());

        // List<Document> topDocuments = vectorDb.query(queryEmbedding, 5); // 前N条相似数据
        // ChatCompletionUserMessageParam.Content.ofText("");
        // ChatCompletionAssistantMessageParam chatCompletionAssistantMessageParam =
        // ChatCompletionAssistantMessageParam
        // .builder().content("").build();
        // user → assistant → user → assistant
        // [
        // { "role": "system", "content": "你是我的私人AI助手" },
        // { "role": "user", "content": "东京有什么景点？" },
        // { "role": "assistant", "content": "你可以去东京塔、浅草寺……" },
        // { "role": "user", "content": "适合小孩子的有哪些？" }
        // ]
        // [
        // { "role": "system", "content": "请对以下对话内容进行简要总结，以便后续继续对话。" },
        // { "role": "user", "content": "今天天气怎么样？\n东京今天晴，气温20度。\n那我该穿什么？\n推荐穿薄外套。\n..."
        // }
        // ]
        // 2. 构建请求
        ChatCompletionCreateParams.Builder createParamsBuilder = ChatCompletionCreateParams.builder()
                .model(ChatModel.GPT_4) // or GPT_3_5_TURBO
                .maxCompletionTokens(1024)
                .tools(tools)
                .n(1)
                // .addMessage(chatCompletionAssistantMessageParam)
                // .addSystemMessage("你是一个知识丰富、表达清晰的智能助手，请用简体中文回答用户问题。") // 设定补充
                .addUserMessage(question); // 依据 某些信息 回答

        // builder.addMessage(ChatCompletionMessage.ofSystem("以下是相关背景信息：\n" +
        // ragContext));

        // 3. 执行第一次请求
        ChatCompletion chatCompletion = openAIClient.chat().completions().create(createParamsBuilder.build());

        // 4. 处理返回内容（包括 tool call 或 content）
        Optional<ChatCompletionMessage> messageOpt = chatCompletion.choices().stream()
                .map(ChatCompletion.Choice::message)
                .peek(createParamsBuilder::addMessage)
                .findFirst();

        String finalAnswer = "对不起，未能生成回答。";

        if (messageOpt.isPresent()) {
            ChatCompletionMessage message = messageOpt.get();

            // ✅ 没有调用 Tool，直接返回原始 content
            if (message.toolCalls().isEmpty()) {
                finalAnswer = message.content().orElse("对不起，未获取回答内容");
                System.out.println("🔹 AI 回复内容：" + finalAnswer);
            } else {
                // ✅ 存在 Tool 调用，逐个执行
                for (ChatCompletionMessageToolCall toolCall : message.toolCalls().get()) {
                    String toolResult = callFunction(toolCall.function());

                    // 将 Tool 执行结果作为系统消息加入上下文
                    createParamsBuilder.addMessage(ChatCompletionToolMessageParam.builder()
                            .toolCallId(toolCall.id())
                            .content(toolResult)
                            .build());
                }

                // 🔁 发起第二轮请求，基于工具执行后的上下文
                ChatCompletion finalCompletion = openAIClient.chat().completions()
                        .create(createParamsBuilder.build());

                finalAnswer = finalCompletion.choices().get(0).message().content()
                        .orElse("对不起，工具执行后仍未获取回答内容");

                System.out.println("🔹 Tool 执行后的 AI 回复内容：" + finalAnswer);
            }
        }

        return ResponseEntity.ok(Map.of("answer", finalAnswer));
    }

    private static final List<String> WAKE_WORDS = List.of("小度", "小度小度", "小爱", "siri", "hey siri", "hi 小度", "你好小度");

    private boolean containsWakeWord(String input) {
        if (input == null)
            return false;

        // 去除标点、空格后统一转小写
        String normalized = input.replaceAll("[\\s,.，。！？!?、~·]", "").toLowerCase();

        return WAKE_WORDS.stream().anyMatch(wakeWord -> normalized.contains(wakeWord.toLowerCase()));
    }

    private static String callFunction(ChatCompletionMessageToolCall.Function function) {
        switch (function.name()) {
            case "get-current-time" -> {
                ZonedDateTime now = ZonedDateTime.now(ZoneId.of("Asia/Tokyo"));
                DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy年MM月dd日 HH:mm:ss E", Locale.JAPAN);
                return "现在的时间是：" + now.format(formatter) + "。";
            }

            case "get-sdk-quality" -> {
                JsonValue arguments;
                try {
                    arguments = JsonValue.from(jsonMapper().readTree(function.arguments()));
                } catch (JsonProcessingException e) {
                    throw new IllegalArgumentException("Bad function arguments", e);
                }

                String sdkName = ((JsonObject) arguments).values().get("name").asStringOrThrow();
                if (sdkName.contains("OpenAI")) {
                    return sdkName + " 是一个功能强大且易于集成的 SDK！";
                }
                return sdkName + " 的质量未知，请谨慎使用。";
            }

            default -> throw new IllegalArgumentException("Unknown function: " + function.name());
        }

    }

}
