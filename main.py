import os
from rag_agent import RAGAgent

from config import VECTOR_DB_PATH, MODEL_NAME, prompt_for_api_config


def main():

    if not os.path.exists(VECTOR_DB_PATH):
        return

    # 启动时交互式配置 API Key（环境变量未设置时提示输入）
    prompt_for_api_config()

    # 初始化RAG Agent
    agent = RAGAgent(model=MODEL_NAME)

    # 检查知识库
    count = agent.vector_store.get_collection_count()
    if count == 0:
        return

    agent.chat()


if __name__ == "__main__":
    main()
