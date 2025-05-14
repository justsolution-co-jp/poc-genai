package com.example.java_rag.agent;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

public interface CodeAgent {

    // SystemMessage 说明 ，让 LLM理解 自然语言 并构建 参数 调用接口
    @SystemMessage("""
            你是一个智能 Git 助手。

            你的任务是接收用户输入的自然语言指令，并调用如下工具方法完成操作：
            - GitTool.commitAndPush(List<String> filePaths, String message)

            用户的输入格式举例：
            - “请将 java-rag/src/main/java/com/example/java_rag/agent/CodeAgent.java 提交到远程，提交说明是：测试 Agent 功能。”
            - “提交文件 src/index.js，备注是 修复问题。”

            举例只是参考格式，实际解析要以用户输入自然语言指令为准

            用户提到的文件或者路径作为参数filePaths，可能有多个；用户提到的备注或者提交说明等内容作为参数message

            请直接解析文件路径字符串列表和备注文字，并调用工具方法。
            ⚠️ 不要解释命令格式，也不要只输出格式样例，请直接执行调用。
            记住：一定要调用方法commitAndPush

            """)
    @UserMessage("{input}")
    String chat(@V("input") String input);
}