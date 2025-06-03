package com.example.java_rag.utils;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Paths;

public class TssTool {

    public String callTTS(String text, String languageId) {
        try {
            HttpClient client = HttpClient.newHttpClient();
            // 谁的声音
            // String speakerWavPath = "/path/to/your/speaker.wav";

            // 构建 URL 参数字符串
            // String textParam = URLEncoder.encode("你好，这是测试语音。", StandardCharsets.UTF_8);
            // String speaker = URLEncoder.encode("Daisy Studious", StandardCharsets.UTF_8);

            String url = String.format(
                    "http://localhost:5002/api/tts?text=%s",
                    text);

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
}
