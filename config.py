import os

#API配置 —— 不在代码中硬编码敏感信息，改为运行时手动配置
#优先级：环境变量 > 启动时交互式输入
#环境变量示例（Windows）:  set OPENAI_API_KEY=sk-xxxxxxxx
#环境变量示例（Linux/Mac）: export OPENAI_API_KEY=sk-xxxxxxxx
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-max").strip()
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-v3").strip()


def prompt_for_api_config() -> None:
    """程序启动时交互式配置 API Key（仅当环境变量未设置时提示输入）。

    输入的 Key 会被写回 os.environ，保证 uvicorn reload 等子进程也能继承。
    """
    global OPENAI_API_KEY
    if OPENAI_API_KEY:
        # 已通过环境变量配置，无需再问
        return
    print("=" * 60)
    print("检测到尚未配置 OPENAI_API_KEY。")
    print("你可以通过以下任一方式配置：")
    print("  1. 设置环境变量后重新运行：")
    print("       Windows : set OPENAI_API_KEY=sk-xxxxx")
    print("       Linux/Mac: export OPENAI_API_KEY=sk-xxxxx")
    print("  2. 现在直接手动输入（仅本次运行有效，不写入任何文件）")
    print("=" * 60)
    try:
        key = input("请输入 OPENAI_API_KEY: ").strip()
    except (EOFError, KeyboardInterrupt):
        key = ""
    if not key:
        raise RuntimeError("未提供 OPENAI_API_KEY，无法启动 RAG 助教。请通过环境变量或在提示时输入。")
    OPENAI_API_KEY = key
    # 写回环境变量，方便后续子进程（如 uvicorn --reload）继承
    os.environ["OPENAI_API_KEY"] = key
    # 出于安全考虑，输入时避免回显
    print("✓ OPENAI_API_KEY 已就绪（本次运行有效）")


# 数据目录配置
DATA_DIR = "./data"

#向量数据库配置
VECTOR_DB_PATH = "./vector_db"
COLLECTION_NAME = "my_rag_collection"

# 文本处理配置
CHUNK_SIZE = 500          # 每段文本长度
CHUNK_OVERLAP = 100       # 两段之间的重叠
MAX_TOKENS = 40000         # 模型最大输出

# RAG配置
TOP_K = 5
BM25_K1 = 1.2
BM25_B = 0.75
RRF_K = 60
