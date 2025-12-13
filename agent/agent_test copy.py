from typing import Annotated, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 1. モデルの指定
llm = ChatOllama(
    base_url="http://host.docker.internal:11434",
    model="rnj-1:8b-instruct-fp16", 
    temperature=0
)

# 2. ツールの定義
@tool
def multiply(a: int, b: int) -> int:
    """2つの整数を掛け算します。"""
    print(f"\n[DEBUG] ツールが呼び出されました: {a} * {b}")
    return a * b

tools = [multiply]
llm_with_tools = llm.bind_tools(tools)

# 3. グラフの状態定義
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 4. ノードの定義
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 5. グラフの構築 (StateGraphを使用)
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# 条件付きエッジ: ツール呼び出しが必要なら tools へ、そうでなければ終了
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# ツール実行後は必ず chatbot に戻る
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

agent_executor = graph_builder.compile()

# 6. 実行
print("--- 実験開始 ---")
query = "25かける45はいくつですか？"
response = agent_executor.invoke({"messages": [("user", query)]})

# 結果の表示
print(f"\n最終回答: {response['messages'][-1].content}")