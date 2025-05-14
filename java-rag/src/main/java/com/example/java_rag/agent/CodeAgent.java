package com.example.java_rag.agent;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

public interface CodeAgent {

    // SystemMessage 说明 ，让 LLM理解 自然语言 并构建 参数 调用接口
    @SystemMessage("""
            你是一个智能 Git 助手。

            你的任务是根据用户输入，自主判断是否需要执行 Git 操作。
            当用户要求你“提交某些文件”并附带“提交说明”时，请调用：

            GitTool.commitAndPush(filePaths: List<String>, message: String)

            - filePaths：提到的所有文件路径
            - message：用户描述的提交说明

            你的输出不应是建议或文字说明，而是直接触发工具函数调用。
            不要输出示例、也不要要求用户重新表述，直接执行操作即可。
            """)
    @UserMessage("{input}")
    String chat(@V("input") String input);

}