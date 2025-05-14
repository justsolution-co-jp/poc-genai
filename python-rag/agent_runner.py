from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from tools import GitCommitTool

llm = ChatOpenAI(temperature=0)  # 需设置 OPENAI_API_KEY 环境变量

agent = initialize_agent(
    tools=[GitCommitTool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def handle_commit_instruction(prompt: str):
    return agent.run(prompt)