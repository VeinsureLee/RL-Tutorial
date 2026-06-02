"""FastAPI app 入口。

启动命令：
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI

from api.routes import status, test, train

app = FastAPI(
    title="MARL-Nav API",
    description="Multi-agent RL navigation algorithm server",
    version="0.1.0",
)
app.include_router(train.router, tags=["train"])
app.include_router(status.router, tags=["status"])
app.include_router(test.router, tags=["test"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
