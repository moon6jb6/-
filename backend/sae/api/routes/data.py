"""数据溯源路由 — 漂移检测、数据谱系"""

from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException

from api.schemas.request import DriftCheckRequest
from api.schemas.response import (
    DataLineageResponse,
    DriftCheckResponse,
    ErrorResponse,
    LineageEdge,
    LineageNode,
)
from api.services.drift_service import compute_drift

router = APIRouter(prefix="/v1/data", tags=["数据溯源"])


@router.post(
    "/drift-check",
    response_model=DriftCheckResponse,
    responses={
        400: {"model": ErrorResponse, "description": "参数错误"},
    },
    summary="数据漂移检测",
    description="检测数据集漂移情况，返回 PSI 指标和告警。",
)
async def drift_check(req: DriftCheckRequest):
    """计算 PSI 并返回漂移检测结果。"""
    try:
        result = compute_drift(
            dataset_id=req.dataset_id,
            reference_period=req.reference_period,
            target_period=req.target_period,
        )
        return DriftCheckResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "drift_error", "message": str(e)}},
        )


@router.get(
    "/lineage/{dataset_id}",
    response_model=DataLineageResponse,
    summary="数据谱系",
    description="获取数据集的完整谱系图（parents, transformations, children）。",
)
async def get_data_lineage(dataset_id: str):
    """返回数据集谱系信息。

    当前使用演示数据，生产环境应从谱系存储中查询。
    """
    # 演示谱系数据
    now = datetime.now(timezone.utc).isoformat()

    parents = [
        LineageNode(id="ds_raw_001", name="原始用户数据", type="raw"),
        LineageNode(id="ds_raw_002", name="外部征信数据", type="raw"),
    ]

    transformations = [
        LineageEdge(
            source="ds_raw_001",
            target=dataset_id,
            transformation="特征工程：归一化 + 缺失值填充",
        ),
        LineageEdge(
            source="ds_raw_002",
            target=dataset_id,
            transformation="数据合并：按 user_id 关联",
        ),
    ]

    children = [
        LineageNode(id="ds_train_001", name="训练集", type="derived"),
        LineageNode(id="ds_test_001", name="测试集", type="derived"),
    ]

    return DataLineageResponse(
        dataset_id=dataset_id,
        parents=parents,
        transformations=transformations,
        children=children,
        created_at=now,
    )
