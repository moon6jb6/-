"""数据溯源路由 — 漂移检测、数据谱系、数据集管理"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.services.data_registry import data_registry
from api.services.drift_service import compute_drift

router = APIRouter(prefix="/v1/data", tags=["数据溯源"])


# ── Request Models ──

class LineageNodeReq(BaseModel):
    dataset_id: str
    name: str

class TransformationReq(BaseModel):
    type: str
    description: str

class LineageRegisterRequest(BaseModel):
    dataset_id: str
    parents: List[LineageNodeReq] = []
    transformations: List[TransformationReq] = []
    children: List[LineageNodeReq] = []

class DatasetRegisterRequest(BaseModel):
    name: str
    source: str
    columns: List[str]
    row_count: int

class DriftCheckBody(BaseModel):
    dataset_id: Optional[str] = None
    reference_period: List[str]
    target_period: List[str]
    reference_file: Optional[str] = None
    target_file: Optional[str] = None


# ── Endpoints ──

@router.post("/drift-check")
async def drift_check(req: DriftCheckBody):
    """检测数据漂移。"""
    result = compute_drift(
        dataset_id=req.dataset_id or "default",
        reference_period=req.reference_period,
        target_period=req.target_period,
        reference_file=req.reference_file,
        target_file=req.target_file,
    )
    return result


@router.get("/lineage/{dataset_id}")
async def get_lineage(dataset_id: str):
    """获取数据集谱系。"""
    lineage = data_registry.get_lineage(dataset_id)
    if lineage is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "not_found", "message": f"数据集 {dataset_id} 不存在"}
        })
    return {"dataset_id": dataset_id, "lineage": lineage}


@router.post("/lineage")
async def register_lineage(body: LineageRegisterRequest):
    """注册/更新数据集谱系。"""
    success = data_registry.set_lineage(
        dataset_id=body.dataset_id,
        parents=[p.dict() for p in body.parents],
        transformations=[t.dict() for t in body.transformations],
        children=[c.dict() for c in body.children],
    )
    if not success:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "not_found", "message": f"数据集 {body.dataset_id} 不存在，请先注册"}
        })
    return {"message": "谱系已更新", "dataset_id": body.dataset_id}


@router.post("/datasets")
async def register_dataset(body: DatasetRegisterRequest):
    """注册新数据集。"""
    result = data_registry.register(
        name=body.name, source=body.source,
        columns=body.columns, row_count=body.row_count,
    )
    return result


@router.get("/datasets")
async def list_datasets():
    """列出所有已注册数据集。"""
    datasets = data_registry.list_datasets()
    return {"datasets": datasets, "total": len(datasets)}


# ── 初始化示例数据 ──

def _init_sample():
    import os
    sae_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_path = os.path.join(sae_root, "task_b_lstm", "aapl_daily.csv")
    if os.path.exists(csv_path) and not data_registry.get("ds_aapl_stock"):
        # 注册父数据集
        if not data_registry.get("ds_yahoo_raw"):
            data_registry.register(
                name="Yahoo Finance Raw Data", source="Yahoo Finance API",
                columns=["date", "open", "high", "low", "close", "adj_close", "volume"],
                row_count=5000,
            )
        ds = data_registry.register(
            name="AAPL Stock Data", source="task_b_lstm/aapl_daily.csv",
            columns=["date", "open", "high", "low", "close", "volume"],
            row_count=1000,
            lineage={
                "parents": [{"dataset_id": "ds_yahoo_raw", "name": "Yahoo Finance Raw Data"}],
                "transformations": [{"type": "etl", "description": "日线OHLCV提取，去除非交易日"}],
                "children": [],
            },
        )
        # 覆盖为固定ID便于引用
        data_registry._datasets["ds_aapl_stock"] = data_registry._datasets.pop(ds["dataset_id"])
        data_registry._datasets["ds_aapl_stock"]["dataset_id"] = "ds_aapl_stock"
        data_registry._save()

_init_sample()
