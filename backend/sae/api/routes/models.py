"""模型管理路由 — 注册、版本查询"""

from fastapi import APIRouter, HTTPException

from api.schemas.request import ModelRegisterRequest
from api.schemas.response import (
    ErrorResponse,
    ModelRegisterResponse,
    ModelVersionsResponse,
)
from api.services.model_registry import model_registry

router = APIRouter(prefix="/v1/models", tags=["模型管理"])


@router.post(
    "/register",
    response_model=ModelRegisterResponse,
    responses={
        400: {"model": ErrorResponse, "description": "参数错误"},
    },
    summary="注册模型",
    description="注册一个新模型到智信引擎平台，生成唯一 model_id。",
)
async def register_model(req: ModelRegisterRequest):
    """注册新模型到内存注册表。"""
    result = model_registry.register(
        name=req.name,
        model_type=req.type,
        framework=req.framework,
        version=req.version,
        file_hash=req.file_hash,
    )
    return ModelRegisterResponse(**result)


@router.get(
    "/{model_id}/versions",
    response_model=ModelVersionsResponse,
    responses={
        404: {"model": ErrorResponse, "description": "模型未找到"},
    },
    summary="模型版本历史",
    description="获取指定模型的所有版本历史。",
)
async def get_model_versions(model_id: str):
    """查询模型版本列表。"""
    result = model_registry.get_versions(model_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "model_not_found", "message": f"模型 {model_id} 未注册"}},
        )
    return ModelVersionsResponse(**result)
