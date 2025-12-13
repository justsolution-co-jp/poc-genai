# Agent Development with GPU and Local LLM

このプロジェクトは、Devcontainer環境でAgentを開発し、ホスト側で動作するLLM (Ollama) と連携するためのサンプル構成です。

## 前提条件

*   Docker Desktop または Docker Engine がインストールされていること。
*   NVIDIA GPUを使用する場合、NVIDIA Container Toolkitがホスト側にインストールされていること。
*   ホストマシンに [Ollama](https://ollama.com/) がインストールされていること。

## セットアップ手順

### 1. ホスト側でのOllamaの設定

Devcontainerからホスト側のOllamaにアクセスするためには、Ollamaが外部からの接続を受け付けるように設定する必要があります。

**Linux / macOS:**
```bash
OLLAMA_HOST=0.0.0.0 ollama serve
```

**Windows (PowerShell):**
```powershell
$env:OLLAMA_HOST="0.0.0.0"; ollama serve
```

### 2. モデルの準備

別のターミナルを開き、使用するモデルをプルします。

```bash
ollama pull rnj-1:8b-instruct-fp16
```
※ `rnj-1:8b-instruct-fp16` が存在しない場合や、別のモデルを使いたい場合は、適宜モデル名を変更してください（例: `qwen2.5:32b`, `llama3` など）。`agent.py` 内の `model_name` も合わせて変更する必要があります。

### 3. Devcontainerの起動

VS Codeでこのフォルダを開き、「Reopen in Container」を実行します。
初回起動時に必要なPythonパッケージが自動的にインストールされます。

### 4. Agentの実行

Devcontainer内のターミナルで以下のコマンドを実行します。

```bash
python agent.py
```

成功すれば、ホスト側のLLMからの応答が表示されます。

## 構成の詳細

*   **Devcontainer**: NVIDIA PyTorchイメージを使用し、GPUリソースにアクセス可能です。
*   **ネットワーク**: `host.docker.internal` を使用してコンテナからホストへ通信します。
*   **ライブラリ**: `openai` ライブラリを使用して、OllamaのOpenAI互換APIと通信します。
