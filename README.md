# 生成AI関連の実証実験

# 🧠 Python-RAG（検索拡張生成：Retrieval-Augmented Generation）

本プロジェクトは、Python を用いて構築された **RAG（検索拡張生成）システム**です。TTS（音声合成）、LoRA による微調整推論、FAQ 質問応答などの機能モジュールが統合されており、中国語環境における音声質問応答やテキストベースの推論タスクに適しています。


## 🧠 Java-RAG（検索拡張生成：Retrieval-Augmented Generation for Java）

本プロジェクトは、Java で構築された **RAG（検索拡張生成）システム**のバックエンド実装です。Agent モジュールや、OpenAI・Ollama など複数の LLM に対応しており、AI チャット・意味検索・情報検索の場面に適しています。


## 💡 LLM アプリ開発の全体像まとめ

このドキュメントは、大規模言語モデル（LLM）を活用した質問応答・RAG・Agent システムの開発者向けに、使用される主要技術スタックや選定理由を整理したものです。


---

### 🧠 1. Python は LLM アプリ開発の中心言語

#### ✅ なぜ Python が中心なのか？

- モデル学習・推論のライブラリはほぼ全て Python 製
- LLM 統合系のツールが非常に充実（LangChain, LlamaIndex など）
- OpenAI / HuggingFace / TTS / Whisper 等の全ての API・SDK が Python ベース

#### 📦 主なフレームワーク

| フレームワーク       | 主な役割                                       |
| -------------------- | ---------------------------------------------- |
| **LangChain**        | LLM 統合・RAG・Agent・ツール連携などの実装支援 |
| **LlamaIndex**       | ドキュメントインデックスと検索・統合の最適化   |
| **Autogen / CrewAI** | 複数 Agent の協調による自動作業処理            |
| **FastAPI / Flask**  | Python 製の軽量 API サーバ構築                 |

#### 🧪 開発例：質問応答システムの基本構成

```text
1. ユーザーの入力：「このファイルの意味は何ですか？」
2. ベクトルDB（例：FAISS / Qdrant）で関連ドキュメントを検索
3. ドキュメント + 質問 をプロンプトにして LLM に送信
4. LLM の出力をそのまま返す
```

> ⚠️ これらの処理を生コードで書くと複雑になり保守も困難になります  
> ✅ LangChain / LlamaIndex を使えば簡単に統合できます

---

### ☕️ 2. Java と LLM の統合：エンタープライズ対応向け

#### ✅ なぜ Java にも価値があるのか？

- エンタープライズ向けの堅牢な API / 認証 / モニタリング機能
- 既存の SpringBoot プロジェクトと統合しやすい
- LangChain4j や Spring AI による LLM 呼び出しが可能

#### 🧰 主なフレームワーク

| フレームワーク      | 機能                             |
| ------------------- | -------------------------------- |
| **Spring AI**       | Spring 向けの LLM 統合モジュール |
| **LangChain4j**     | Java 版 LangChain                |
| **Ollama Java SDK** | ローカル LLM（Mistral 等）の統合 |

#### 🌐 開発構成の一例

- Spring Boot で API 提供
- LangChain4j による RAG & Agent 処理
- フロントエンドからの接続（Next.js など）

---

### 🌐 3. TypeScript / Node.js の活用範囲

#### 🚀 主な活用場面

| 用途             | 技術スタック                |
| ---------------- | --------------------------- |
| フロント UI      | Next.js, React              |
| LLM 連携         | LangChain.js, LlamaIndex.js |
| ツールビューワー | LangSmith, Prompt 工具など  |

#### ❗️制限

- Agent 編成やベクトル検索は Python の方が優れている
- TypeScript 側では LLM を直接呼び出す用途が主

---

## ✅ 言語別の LLM 対応比較

| 特性           | Python                       | Java                 | TypeScript        |
| -------------- | ---------------------------- | -------------------- | ----------------- |
| モデル呼び出し | ✅ 最も豊富なライブラリあり   | ⚠️ API 通信が主       | ⚠️ API 経由        |
| RAG 編成       | ✅ LangChain, LlamaIndex      | ✅ LangChain4j        | ⚠️ 限定的サポート  |
| Agent 機能     | ✅ 多機能                     | ⚠️ 限定的な構成       | ❌ 非対応が多い    |
| エコシステム   | ✅ AI モデル、TTS、Whisper 等 | ✅ API / モニタリング | ✅ フロント技術群  |
| 適した用途     | AI 推論、Agent、RAG          | API 提供、業務統合   | UI 開発、Web 連携 |

---

## ✅ 開発アプローチのまとめ

- 🧠 モデル実行・Agent を含む LLM 活用 → **Python**
- 🏢 API 統合・既存 Java システム連携 → **Java（Spring AI + LangChain4j）**
- 🌐 フロント UI & クラウド連携 → **TypeScript + バックエンド連携**




# 🧠 RAG（Retrieval-Augmented Generation）開発フロー

RAG は「検索によって補強された生成モデル」です。LLM 単体ではなく、外部知識（ドキュメントなど）と組み合わせて高精度な回答を得るためのアーキテクチャです。

---

## 🚀 全体フロー

1. **ユーザー入力**
   - 例：「この製品の保証はどれくらいですか？」

2. **ドキュメント検索（Retrieval）**
   - 質問をベクトル化
   - ベクトルデータベース（例：FAISS, Qdrant）から関連文書チャンクを検索

3. **プロンプト構築（Prompt Engineering）**
   - 検索結果 + ユーザー質問 を組み合わせたプロンプトを作成
   - 例：
     ```
     以下の情報に基づき、質問に答えてください：
     ---
     [関連文書チャンクA]
     [関連文書チャンクB]
     ---
     Q: この製品の保証はどれくらいですか？
     ```

4. **LLM 呼び出し**
   - OpenAI, Ollama, Claude などにプロンプトを渡す
   - 回答を取得

5. **回答返却**
   - ユーザーに自然な形式で表示
   - オプションで音声合成や UI 表示を連携

---

## 🔧 開発に使えるツール／ライブラリ

- **LangChain / LangChain4j**
- **LlamaIndex**
- **FAISS / Qdrant / Weaviate**
- **OpenAI API / HuggingFace Transformers / Ollama**
- **Flask / FastAPI / Spring Boot**

---

## 💡 応用

- FAQ Bot
- 法律・金融など専門知識QA
- 社内ナレッジ検索


# 🧠 RAG（Retrieval-Augmented Generation）開発フロー

RAG は「検索によって補強された生成モデル」です。LLM 単体ではなく、外部知識（ドキュメントなど）と組み合わせて高精度な回答を得るためのアーキテクチャです。

---

## 🚀 全体フロー

1. **ユーザー入力**
   - 例：「この製品の保証はどれくらいですか？」

2. **ドキュメント検索（Retrieval）**
   - 質問をベクトル化
   - ベクトルデータベース（例：FAISS, Qdrant）から関連文書チャンクを検索

3. **プロンプト構築（Prompt Engineering）**
   - 検索結果 + ユーザー質問 を組み合わせたプロンプトを作成
   - 例：
     ```
     以下の情報に基づき、質問に答えてください：
     ---
     [関連文書チャンクA]
     [関連文書チャンクB]
     ---
     Q: この製品の保証はどれくらいですか？
     ```

4. **LLM 呼び出し**
   - OpenAI, Ollama, Claude などにプロンプトを渡す
   - 回答を取得

5. **回答返却**
   - ユーザーに自然な形式で表示
   - オプションで音声合成や UI 表示を連携

---

## 🔧 開発に使えるツール／ライブラリ

- **LangChain / LangChain4j**
- **LlamaIndex**
- **FAISS / Qdrant / Weaviate**
- **OpenAI API / HuggingFace Transformers / Ollama**
- **Flask / FastAPI / Spring Boot**

---

## 💡 応用

- FAQ Bot
- 法律・金融など専門知識QA
- 社内ナレッジ検索


# 🤖 Agent 開発フロー（LLM Tool Agent）

Agent はツール実行を自動で判断できる「思考型」AI です。ユーザーの質問に応じて、必要な処理（API, DB, ファイル操作など）を自律的に呼び出します。

---

## 🧭 基本フロー

1. **ユーザー入力**
   - 例：「今日の東京の天気は？」

2. **思考ステップ生成**
   - LLM が「何をするべきか」を判断（思考プロンプト）

3. **ツール選定**
   - 登録済みのツールリストから、必要なものを選択
   - 例：`get_current_weather(city="Tokyo")`

4. **ツール実行**
   - 外部 API 呼び出しなどを実行
   - 結果を LLM に渡す

5. **最終回答生成**
   - 実行結果をもとに自然な文章を生成して返答

---

## ⚙️ 技術構成

- LangChain / LangChain4j
- AgentExecutor / initialize_agent
- Tool定義：Pythonの関数 / Javaのクラス
- LLM：OpenAI, Ollama, Claude など

---

## 🧪 よくある Agent ツール例

- 現在時刻取得
- コード修正ツール
- Git 操作自動化（コミット、Push）
- DB 検索
- Web スクレイピング
- 自然言語からのSQL生成

---

## 📦 応用

- GitHub Copilot 風のコーディング支援
- AI IT サポート窓口
- 複雑なマルチステップワークフロー自動化


