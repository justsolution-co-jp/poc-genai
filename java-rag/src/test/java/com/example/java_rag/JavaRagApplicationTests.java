package com.example.java_rag;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class JavaRagApplicationTests {
	// private final HttpClient client = HttpClient.newHttpClient();

	@Test
	void callChatGPT() {
		String OPENAI_API_KEY = System.getenv("OPENAI_API_KEY");
		// HttpRequest request = HttpRequest.newBuilder()

		// .uri(URI.create("https://api.openai.com/v1/models"))
		// .header("Authorization", "Bearer " + OPENAI_API_KEY)
		// // .header("Content-Type", "application/json")
		// .build();

		// HttpResponse<String> response;
		// try {
		// response = client.send(request, HttpResponse.BodyHandlers.ofString());
		// System.out.println("💬 ChatGPT 返回原始内容：" + response.body());
		// } catch (IOException e) {
		// System.out.println(e);
		// } catch (InterruptedException e) {
		// System.out.println(e);
		// }

	}

}
