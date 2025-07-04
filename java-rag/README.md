# 🧠 Java-RAG（検索拡張生成：Retrieval-Augmented Generation）

このプロジェクトは、Java + Spring Boot による RAG（検索拡張生成）システムの実装です。OpenAI や Ollama モデルの統合、LangChain4j によるベクトル検索、複数エージェント機能を提供し、AI チャットや情報検索の基盤として活用できます。

---

## 📁 プロジェクト構成

| パス / クラス | 説明 |
|---------------|------|
| `agent/CodeAgent.java` | コード操作用 AI エージェント |
| `agent/DateTimeAgent.java` | 日時取得エージェント |
| `config/OllamaConfig.java` | Ollama モデルの設定 |
| `config/OpenAiConfig.java` | OpenAI API 設定 |
| `controller/OpenAiController.java` | OpenAI チャット関連 API |
| `controller/RagController.java` | RAG 処理と TTS（音声合成）API |
| `service/LangchainRagService.java` | LangChain4j を用いた検索処理サービス |
| `utils/DateTimeTool.java` | 日時ユーティリティ |
| `utils/GitTool.java` | Git 操作用ツール |
| `utils/PromptUtil.java` | プロンプト補助ツール |
| `utils/TssTool.java` | 音声合成処理補助ツール |
| `JavaRagApplication.java` | Spring Boot アプリケーションエントリーポイント |

---

## 🛠 使用技術

- Java 17+
- Spring Boot 3.x
- LangChain4j
- OpenAI / Ollama
- Maven
- RESTful API

---

## 🚀 起動手順

1. Java 17+ と Maven をインストール済みであることを確認
2. `.env` または環境変数に `OPENAI_API_KEY` を設定
3. 以下のコマンドで起動：

```bash
./mvnw spring-boot:run
```

起動後、`http://localhost:8080` でエンドポイントが利用可能になります。

---

## 🔗 主なAPIエンドポイント

### `/openAi/chat`  
- OpenAIモデル（GPT-4など）による通常チャット  
- POST JSON 例：
```json
{
  "question": "東京の今日の天気は？"
}
```

---

### `/openAi/chatWithAudio`  
- テキストチャットの結果に加え、音声合成結果も対応予定

---

### `/openAi/chatWithAgent`  
- Tool（日時など）を呼び出せるエージェントチャット

---

### `/rag/ask`  
- LangChain4j + OpenAI による RAG 検索回答

---

### `/rag/generate`  
- TTS（音声合成）API  
- POST JSON：
```json
{
  "text": "こんにちは、これはテストです",
  "language_id": "ja"
}
```

---

### `/rag/audio?path=xxx.wav`  
- 音声ファイルのダウンロード/再生

---

## 📌 補足

- Python ベースのベクトル検索エンジン（RAG）と連携する場合は、`http://localhost:8888/retrieval` にアクセスできる Python API が必要です。
- 音声合成 API は `http://localhost:5002/api/tts` を想定しています。

---

開発やデモ用にカスタマイズしやすい構成になっており、Java での AI 統合学習にも最適です。
---

## 🌐 フロントエンド（HTML）画面

`resources/static/` ディレクトリに以下の HTML ファイルが用意されています。これらのファイルはブラウザ上でインターフェースとして利用可能です。

| ファイル名            | 説明                     |
|----------------------|--------------------------|
| `index.html`         | トップページ／サンプル画面 |
| `speek1.html`        | 音声質問デモ画面（例1）     |
| `speek_upload.html`  | 音声ファイルアップロード画面 |

起動後、これらの画面は `http://localhost:8080/index.html` 等でアクセス可能です。
