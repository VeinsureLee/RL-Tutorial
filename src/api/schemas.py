"""FastAPI 请求/响应 Pydantic 模型。"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    algorithm: str = Field(..., description="Algorithm name from ALGORITHM_REGISTRY")
    map_file: str = Field(default="default")
    config_overrides: dict = Field(default_factory=dict)
    tag: str = Field(default="")


class TestRequest(BaseModel):
    algorithm: str
    model_path: str
    map_file: str = "default"
    max_steps: int = 500


class RunIdResponse(BaseModel):
    run_id: str


class StatusResponse(BaseModel):
    run_id: str
    status: Literal["running", "completed", "failed"]
    episode: int
    total_episodes: int
    latest_reward: float
    model_path: Optional[str] = None
    error: Optional[str] = None


class TestResponse(BaseModel):
    success: bool
    steps: int
    total_reward: float
    gif_path: str
    png_path: str
