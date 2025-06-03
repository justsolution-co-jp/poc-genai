package com.example.java_rag.utils;

import java.util.List;

import com.github.pemistahl.lingua.api.Language;
import com.github.pemistahl.lingua.api.LanguageDetector;
import com.github.pemistahl.lingua.api.LanguageDetectorBuilder;

public class PromptUtil {
    private static final LanguageDetector detector = LanguageDetectorBuilder.fromAllLanguages().build();

    public static Language checkLanguage(String question) {
        return detector.detectLanguageOf(question);
    }

    public static String getTTSLanguage(String question) {
        Language lang = checkLanguage(question);
        String langName = "zh-cn";
        switch (lang) {
            case JAPANESE -> langName = "jp";
            // case CHINESE -> langHint = "请用中文回答问题：";
            default -> langName = "zh-cn";
        }
        return langName;
    }

    public static String buildPromptWithLanguageHint(List<String> chunks, String question) {
        Language lang = checkLanguage(question);

        System.out.println("🌐 识别到提问语言: " + lang.name());

        String langHint = "";
        // switch (lang) {
        // case JAPANESE -> langHint = "質問に日本語で答えてください。";
        // case CHINESE -> langHint = "请用中文回答问题：";
        // default -> langHint = "Please answer the question in English:";
        // }

        String content = String.join("\n\n", chunks);
        return content + "\n\n" + langHint + "\n\n question:" + question;
    }
}
