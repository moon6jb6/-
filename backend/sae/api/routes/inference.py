"""推理溯源路由 — POST /v1/inference/trace"""

from fastapi import APIRouter, HTTPException

from api.schemas.request import InferenceTraceRequest
from api.schemas.response import ErrorResponse, InferenceTraceResponse
from api.services.inference_service import run_inference_trace

router = APIRouter(prefix="/v1/inference", tags=["推理溯源"])


@router.post(
    "/trace",
    response_model=InferenceTraceResponse,
    responses={
        400: {"model": ErrorResponse, "description": "参数错误"},
        404: {"model": ErrorResponse, "description": "模型未找到"},
        500: {"model": ErrorResponse, "description": "服务器内部错误"},
    },
    summary="推理溯源",
    description="提交一次推理溯源请求，返回 SHAP/IG 归因分析结果。",
)
async def inference_trace(req: InferenceTraceRequest):
    """执行推理溯源：检测模型架构 → 路由归因方法 → 返回结果。"""
    try:
        result = run_inference_trace(
            model_id=req.model_id,
            features=req.input_data.features,
            methods=req.methods,
        )
        return InferenceTraceResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "model_not_found", "message": str(e)}},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "internal_error", "message": str(e)}},
        )
