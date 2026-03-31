from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# 加载 .env 中的 OpenAI 配置
load_dotenv()

from app.rag_pipeline import AAPL10KRAG

global_rag_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_rag_system
    # 在应用启动时初始化 RAG (包括模型加载、向量库构建检查等)
    global_rag_system = AAPL10KRAG(data_path="./data/aapl_10k.json")
    yield
    # 清理逻辑

app = FastAPI(title="AAPL 10-K RAG API", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if not global_rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        answer = global_rag_system.ask(request.query)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
