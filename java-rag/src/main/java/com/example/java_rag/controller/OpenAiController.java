package com.example.java_rag.controller;

import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.example.java_rag.config.OpenAiConfig;
import com.example.java_rag.utils.PromptUtil;
import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.models.ChatModel;
import com.openai.models.FunctionDefinition;
import com.openai.models.FunctionParameters;
import com.openai.models.chat.completions.ChatCompletion;
import com.openai.models.chat.completions.ChatCompletionCreateParams;
import com.openai.models.chat.completions.ChatCompletionFunctionCallOption;
import com.openai.models.chat.completions.ChatCompletionTool;
import com.openai.models.models.Model;
import com.openai.models.responses.FunctionTool;
import com.openai.services.blocking.ModelService;

import dev.ai4j.openai4j.chat.ChatCompletionRequest;

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

        // openAIClient.models().list().getData().forEach(model -> {
        // System.out.println("✅ 模型ID: " + model.id());
        // });
        List<String> modelList = openAIClient.models().list().items().stream().map(Model::id).toList();
        // ChatCompletionCreateParams chatParams =
        // ChatCompletionCreateParams.builder().addUserMessage("你好")
        // .model(ChatModel.GPT_4O).build();
        // ChatCompletion chatCompletion =
        // openAIClient.chat().completions().create(chatParams);
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
                "response", content));
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
                "response", content.isPresent() ? content.get() : ""));
    }

}
