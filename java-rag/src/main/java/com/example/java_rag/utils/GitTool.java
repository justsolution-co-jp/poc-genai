package com.example.java_rag.utils;

import java.io.File;
import java.util.List;

import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.lib.ObjectId;
import org.eclipse.jgit.transport.PushResult;
import org.eclipse.jgit.transport.RefSpec;
import org.eclipse.jgit.transport.RemoteRefUpdate;
import org.eclipse.jgit.transport.UsernamePasswordCredentialsProvider;

import dev.langchain4j.agent.tool.Tool;

public class GitTool {

    private static final String DEFAULT_REPO_PATH = "/home/just/projects/poc-genai/";

    @Tool
    public String commitAndPush(List<String> filePaths, String message) {
        try (Git git = Git.open(new File(DEFAULT_REPO_PATH))) {
            // git.add().addFilepattern(".").call();

            for (String filePath : filePaths) {
                git.add().addFilepattern(filePath).call();
            }

            git.commit().setMessage(message).call();
            // git.push().call();

            String username = System.getenv("GITHUB_USERNAME");
            String token = System.getenv("GITHUB_TOKEN");

            String branch = git.getRepository().getBranch();
            Iterable<PushResult> results = git.push()
                    .setRemote("origin")
                    .setRefSpecs(new RefSpec(branch + ":" + branch))
                    .setCredentialsProvider(new UsernamePasswordCredentialsProvider(username, token))
                    .call();

            ObjectId head = git.getRepository().resolve("HEAD");
            System.out.println("🔍 当前 HEAD commit: " + head.getName());

            // ✅ 检查 push 是否成功
            for (PushResult result : results) {
                System.out.println(result.getMessages());
                for (RemoteRefUpdate update : result.getRemoteUpdates()) {
                    RemoteRefUpdate.Status status = update.getStatus();
                    String remoteName = update.getRemoteName();

                    System.out.println("🚀 Push to " + remoteName + " => " + status + " => " + update.getMessage());

                    if (status != RemoteRefUpdate.Status.OK &&
                            status != RemoteRefUpdate.Status.UP_TO_DATE) {
                        return "❌ 推送失败：分支 " + remoteName + " 状态为 " + status;
                    }
                }
            }

            System.out.println("当前分支是：" + git.getRepository().getBranch());

            return "推送成功";
        } catch (Exception e) {
            System.out.println("🔥 推送出错！");
            e.printStackTrace();
            return "Git 错误: " + e.getMessage();
        }
    }
}
