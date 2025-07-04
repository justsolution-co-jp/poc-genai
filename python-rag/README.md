# 🧠 Python-RAG（検索拡張生成：Retrieval-Augmented Generation）

本プロジェクトは、Python を用いて構築された **RAG（検索拡張生成）システム**です。TTS（音声合成）、LoRA による微調整推論、FAQ 質問応答などの機能モジュールが統合されており、中国語環境における音声質問応答やテキストベースの推論タスクに適しています。

---

## 📁 プロジェクト構成

| ファイル/ディレクトリ | 説明                                       |
| --------------------- | ------------------------------------------ |
| `main.py`             | メインエントリーポイント。RAG 流れの実行用 |
| `rag_engine.py`       | 検索と生成の中核ロジック                   |
| `infer_phi_lora.py`   | Phi モデルの読み込みと LoRA 推論           |
| `train_phi_lora.py`   | LoRA によるモデル微調整スクリプト          |
| `safe_speak.py`       | セーフスピーク機能（フィルター処理など）   |
| `transcribe_mp3.py`   | 音声ファイルからテキストへの変換（ASR）    |
| `tts_demo.py`         | テキストから音声への変換（TTS）            |
| `agent_runner.py`     | エージェント管理・指令の制御モジュール     |
| `server.py`           | サーバー起動スクリプト（API 提供可能）     |
| `tools.py`            | ユーティリティ関数群                       |
| `csv2txt.py`          | FAQ CSV をテキスト形式に変換               |
| `faqs.csv`            | FAQ データベース（構造化CSV）              |
| `document.txt`        | ベクトル検索用のテキストコーパス           |
| `requirements.txt`    | Python パッケージ依存リスト                |
| `README.md`           | 本ドキュメント                             |
| `tts-zh.log`          | 音声合成に関するログファイル               |
| `checkpoint-2053/`    | LoRA モデルのチェックポイント保存先        |
| `venv/`               | Python 仮想環境ディレクトリ（開発用）      |

---


## 📦 前提：Qdrant・Ollama のインストール

本プロジェクトを実行する前に、以下のサービスをローカルまたはクラウド環境にセットアップしてください：

### ✅ Qdrant（ベクトルデータベース）

Docker での起動例：

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Qdrant 管理画面（オプション）：  
http://localhost:6333/

### ✅ Ollama（ローカル大規模言語モデル）

Ollama インストール方法（Mac/Linux/WSL）：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

モデル例（mistral をダウンロード）：

```bash
ollama run mistral
```

モデル管理画面：  
http://localhost:11434/


## 🚀 クイックスタート（Python仮想環境の利用推奨）

### 1. 仮想環境の作成
```bash
python3 -m venv venv
```

### 2. 仮想環境を有効化

- Windows の場合：
```bash
venv\Scripts\activate
```

- macOS / Linux の場合：
```bash
source venv/bin/activate
```

### 3. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 4. メインプログラムの起動
```bash
python main.py
```

---

## 🛰️ API インターフェース

このバックエンドは Flask を使用して構築されており、以下のエンドポイントが提供されています：

### 🔧 `/agent` - Git 自動コミットエージェント

- **URL**: `/agent`  
- **メソッド**: `POST`  
- **説明**: Git コミットおよび必要に応じて push を自動化します。LangChain の Tool 機能を通じて、自然言語から Git 操作を誘導します。

#### 🔸 リクエスト例（JSON）:

```json
{
  "input": "请提交 src/utils/PromptUtil.java 文件，提交说明为：优化提示模板处理逻辑，推送到远程仓库"
}
```

#### 🔹 レスポンス例:

```json
{
  "result": {
    "output": "已提交并成功推送 ✅"
  }
}
```

---

### 📖 `/retrieval` - RAG文档検索インターフェース

- **URL**: `/retrieval`  
- **メソッド**: `POST`  
- **説明**: RAG モデルにより、事前インデックスされた文書から質問に関連するチャンクを返します。

#### 🔸 リクエスト例（JSON）:

```json
{
  "question": "Pythonで仮想環境を作成するには？"
}
```

#### 🔹 レスポンス例:

```json
{
  "chunks": [
    "仮想環境を作るには python3 -m venv venv を使います。",
    "仮想環境を有効化するには source venv/bin/activate を実行します。"
  ]
}
```

---

### 🔊 `/transcribe` - 音声文字起こし（Whisper）

- **URL**: `/transcribe`  
- **メソッド**: `POST`  
- **説明**: `.wav` 形式の音声ファイルをアップロードし、中国語で音声認識を実行します。

#### 🔸 フォームデータ:

- `file`: `.wav` 形式の音声ファイル

#### 🔹 レスポンス例:

```json
{
  "text": "你好，我是语音识别助手。"
}
```

---

## 🚀 起動と確認

サーバー起動後、以下の URL でエンドポイントを確認できます：

- http://localhost:8888/agent
- http://localhost:8888/retrieval
- http://localhost:8888/transcribe

Postman、curl、または前端からの API リクエストで呼び出し可能です。
