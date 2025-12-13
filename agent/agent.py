import os
from openai import OpenAI

# ホスト側のLLMサーバーのエンドポイント
# Devcontainerからホストへは host.docker.internal でアクセス可能
# Ollamaのデフォルトポートは11434
BASE_URL = "http://host.docker.internal:11434/v1"
API_KEY = "ollama" # OllamaはAPIキーをチェックしないが、ライブラリの仕様上必要

def main():
    print(f"Connecting to LLM at {BASE_URL}...")
    
    try:
        client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY
        )

        # モデル名 (ユーザー指定)
        model_name = "rnj-1:8b-instruct-fp16"

        print(f"Sending request to model: {model_name}")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Hello! Can you describe what you are capable of?"}
            ],
            stream=False
        )

        print("\nResponse from LLM:")
        print("-" * 40)
        print(response.choices[0].message.content)
        print("-" * 40)

    except Exception as e:
        print(f"\nError communicating with LLM: {e}")
        print("\nPlease ensure that:")
        print("1. Ollama is running on your host machine.")
        print(f"2. You have pulled the model: ollama pull {model_name}")
        print("3. Ollama is listening on all interfaces or specifically configured to allow connections from Docker.")
        print("   (e.g. OLLAMA_HOST=0.0.0.0)")

if __name__ == "__main__":
    main()
