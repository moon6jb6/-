"""模型管理路由 — 注册、列表、版本查询、删除"""

import hashlib
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, File, UploadFile

from api.schemas.request import ModelRegisterRequest
from api.schemas.response import (
    ErrorResponse,
    ModelRegisterResponse,
    ModelVersionsResponse,
    ModelListResponse,
)
from api.services.model_registry import model_registry

router = APIRouter(prefix="/v1/models", tags=["模型管理"])

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")


@router.post("/register", response_model=ModelRegisterResponse)
async def register_model(
    req: ModelRegisterRequest,
    model_file: Optional[UploadFile] = File(None),
):
    """注册新模型。可选上传模型文件以自动计算 SHA256。"""
    file_hash = req.file_hash or ""
    sha256_verified = False

    if model_file is not None:
        file_bytes = await model_file.read()
        computed_hash = "sha256:" + hashlib.sha256(file_bytes).hexdigest()
        if req.file_hash and req.file_hash != computed_hash:
            raise HTTPException(status_code=400, detail={
                "error": {
                    "code": "hash_mismatch",
                    "message": "SHA256 不匹配",
                    "details": {"provided": req.file_hash, "computed": computed_hash},
                }
            })
        file_hash = computed_hash
        sha256_verified = True
        os.makedirs(MODELS_DIR, exist_ok=True)
        save_path = os.path.join(MODELS_DIR, f"{uuid.uuid4().hex[:8]}.bin")
        with open(save_path, "wb") as f:
            f.write(file_bytes)

    result = model_registry.register(
        name=req.name, model_type=req.type, framework=req.framework,
        version=req.version, file_hash=file_hash, sha256_verified=sha256_verified,
    )
    return ModelRegisterResponse(**result)


@router.get("", response_model=ModelListResponse)
async def list_models():
    """列出所有已注册模型。"""
    models = model_registry.list_models()
    return {"models": models, "total": len(models)}


@router.get("/{model_id}/versions", response_model=ModelVersionsResponse)
async def get_model_versions(model_id: str):
    """获取模型版本历史。"""
    result = model_registry.get_versions(model_id)
    if result is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "model_not_found", "message": f"模型 {model_id} 未注册"}
        })
    return ModelVersionsResponse(**result)


@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """删除模型。"""
    if not model_registry.delete(model_id):
        raise HTTPException(status_code=404, detail={
            "error": {"code": "model_not_found", "message": f"模型 {model_id} 不存在"}
        })
    return {"message": f"模型 {model_id} 已删除"}


import uuid  # noqa: E402
