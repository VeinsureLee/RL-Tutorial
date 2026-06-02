"""GET /status/{run_id}：查询训练进度。"""
from fastapi import APIRouter, HTTPException

from api.runs import get_run
from api.schemas import StatusResponse

router = APIRouter()


@router.get("/status/{run_id}", response_model=StatusResponse)
def get_status(run_id: str) -> StatusResponse:
    state = get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    return StatusResponse(
        run_id=state.run_id,
        status=state.status,
        episode=state.episode,
        total_episodes=state.total_episodes,
        latest_reward=state.latest_reward,
        model_path=state.model_path,
        error=state.error,
    )
