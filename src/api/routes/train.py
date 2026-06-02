"""POST /train：提交训练任务，立即返回 run_id。"""
from fastapi import APIRouter

from api.runs import start_training
from api.schemas import RunIdResponse, TrainRequest

router = APIRouter()


@router.post("/train", response_model=RunIdResponse)
async def submit_train(req: TrainRequest) -> RunIdResponse:
    run_id = await start_training(
        req.algorithm, req.map_file, req.config_overrides, req.tag
    )
    return RunIdResponse(run_id=run_id)
