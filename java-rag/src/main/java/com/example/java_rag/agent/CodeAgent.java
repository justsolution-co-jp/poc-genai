package com.example.java_rag.agent;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

public interface CodeAgent {

    @UserMessage("用户说：{{it}}")
    @SystemMessage("""
            你是一个代码管理助手，负责执行用户的 Git 和文件操作请求。

            你可以访问以下工具：

            - GitTool.pushToRemote(repoPath): 推送代码到远程仓库。

            ⚠️ 当用户请求 “提交代码”、“推送”、“保存更改” 等操作时，请一定使用对应工具，而不是返回建议或解释。
            """)
    String chat(@V("it") String input);
}