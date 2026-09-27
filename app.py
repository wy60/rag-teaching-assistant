from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_agent import RAGAgent
from fastapi.responses import StreamingResponse, FileResponse
from config import prompt_for_api_config
"""
python process_data.py
uvicorn app:app --reload --port 8000
open frontend/index.html
"""

# 启动时交互式配置 API Key（环境变量未设置时提示输入）
prompt_for_api_config()

# 初始化 RAG 助教
agent = RAGAgent()

# FastAPI 初始化
app = FastAPI()

# 允许任何前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求格式
class ChatRequest(BaseModel):
    query: str
    history: list = []

class ChatResponse(BaseModel):
    answer: str

# POST /chat
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer = agent.answer_question(req.query, chat_history=req.history)
    return ChatResponse(answer=answer)

@app.post("/chat_stream")
def chat_stream(req: ChatRequest):
    def gen():
        for delta in agent.answer_question_stream(req.query, chat_history=req.history):
            yield delta
    return StreamingResponse(gen(), media_type="text/plain")

@app.get("/")
def index():
    return FileResponse("frontend/index.html")
