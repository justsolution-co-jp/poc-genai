import os
from typing import Annotated, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 1. モデルの指定
# ※ コード解析には文脈理解力が必要なため、llama3.1やqwen2.5などのモデル推奨ですが、
#    設定済みの rnj-1 で動作するようにしています。
llm = ChatOllama(
    base_url="http://host.docker.internal:11434",  # 環境に合わせて変更してください (例: http://localhost:11434)
    model="rnj-1:8b-instruct-fp16",
    temperature=0
)

# 2. ツールの定義 (ファイル読み込み用)
@tool
def read_file(file_path: str) -> str:
    """
    指定されたパスのファイルの内容を読み込みます。
    Pythonファイルやテキストファイルを読み込む際に使用してください。
    """
    print(f"\n[DEBUG] ファイル読み込みツール実行: {file_path}")
    
    if not os.path.exists(file_path):
        return f"エラー: ファイル '{file_path}' が見つかりません。"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 内容が長すぎる場合の対策（トークン数節約のため、必要に応じて制限）
            if len(content) > 10000:
                return content[:10000] + "\n...(以下略)..."
            return content
    except Exception as e:
        return f"エラー: ファイル読み込み中に問題が発生しました。 {e}"

tools = [read_file]
llm_with_tools = llm.bind_tools(tools)

# 3. グラフの状態定義
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 4. ノードの定義
def chatbot(state: State):
    # システムプロンプトを会話の冒頭に差し込む（オプションですが精度向上のため）
    # ※ LangGraphのadd_messagesがうまく処理してくれるため、単純にinvokeします
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 5. グラフの構築
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# 条件付きエッジ
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

agent_executor = graph_builder.compile()

# 6. 実行
print("--- Agent実行開始 ---")

# Agentへの指示
query = "現在のディレクトリにある 'agent.py' を読み込んで、このプログラムがどのような機能を持っているか、概要を要約して教えてください。"

# 最初のメッセージとしてシステムプロンプト（役割定義）を入れるとより安定します
initial_state = {
    "messages": [
        SystemMessage(content="あなたは優秀なコードレビューAIです。提供されたツールを使用してコードを読み、その内容を正確に解説してください。"),
        ("user", query)
    ]
}

response = agent_executor.invoke(initial_state)

# 結果の表示
print(f"\n=== 最終回答 ===\n{response['messages'][-1].content}")