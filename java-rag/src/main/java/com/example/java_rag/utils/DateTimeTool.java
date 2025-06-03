package com.example.java_rag.utils;

import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import dev.langchain4j.agent.tool.Tool;

public class DateTimeTool {

    @Tool(name = "queryDate")
    public String queryDate() {
        System.out.println(">>>>>>>>>>>>>>>getDate");
        ZonedDateTime nowInTokyo = ZonedDateTime.now(ZoneId.of("Asia/Tokyo"));
        return nowInTokyo.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss z"));
    }
}
