from flask import Flask, request, jsonify
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama
from rag_engine import RAGEngine, git_commit_and_optional_push
from langchain.agents import Tool
from langchain.agents import initialize_agent, AgentType

app = Flask(__name__)
engine = RAGEngine()

llm = ChatOllama(model="mistral")
tools = [
    Tool(
        name="git_commit_and_optional_push",
        func=git_commit_and_optional_push,
        description="使用 Git 提交代码，并根据内容是否包含 '推送' 决定是否 push 到远程仓库。需要提供 message、path、hint 三个字段。"
    )
]

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个 Git 助手，只能使用工具 `git_commit_and_optional_push` 来提交代码。\n"
     "你必须使用如下格式输出调用工具:\n\n"
     "Action: git_commit_and_optional_push\n"
     "Action Input: {\"message\": \"提交信息\", \"path\": \"文件路径\", \"hint\": \"是否需要推送\"}\n\n"
     "⚠️ 注意：Action Input 必须是合法 JSON，不能是字符串格式，也不能是 Python 风格。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={
        "system_message": (
            "你是一个 Git 工具调用助手，只能使用工具 `git_commit_and_optional_push`。\n"
            "你必须严格按以下格式调用工具，不要添加任何解释或自然语言。\n\n"
            "调用格式必须为：\n"
            "Action: git_commit_and_optional_push\n"
            "Action Input: {\"message\": \"xxx\", \"path\": \"xxx\", \"hint\": \"xxx\"}\n\n"
            "❌ 不要使用 tool_input，不要使用多行或 markdown 格式。\n"
            "✅ 只允许合法的 JSON 对象作为 Action Input。\n"
        )
    }
)


agent_executor = agent

@app.route("/agent", methods=["POST"])
def agent_endpoint():
    data = request.get_json()
    user_input = data.get("input", "")
    if not user_input:
        return jsonify({"error": "请提供 input 字段"}), 400

    try:
        result = agent_executor.invoke({"input": user_input})
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/retrieval", methods=["POST"])
def retrieval():
    data = request.get_json()
    query = data.get("question", "")
    if not query:
        return jsonify({"error": "请提供 question 字段"}), 400
    chunks = engine.retrieve(query)
    return jsonify({"chunks": chunks})

if __name__ == "__main__":
    print("🚀 Flask 启动中...")
    app.run(host="0.0.0.0", port=5002)
