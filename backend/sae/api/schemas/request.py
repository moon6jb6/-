"""Pydantic v2 请求模型 — 智信引擎 API"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── 推理溯源 ────────────────────────────────────────────────────────────

class InputData(BaseModel):
    """推理输入数据"""
    features: Dict[str, Any] = Field(..., description="特征名→值的映射")


class InferenceTraceRequest(BaseModel):
    """POST /v1/inference/trace 请求"""
    model_id: str = Field(..., description="已注册模型ID，如 mdl_xxxxxxxx")
    input_data: InputData = Field(..., description="输入数据")
    methods: List[str] = Field(
        default=["shap"],
        description="归因方法列表，可选: shap, lime, gradcam, ig, tree_shap",
    )


# ── 模型注册 ────────────────────────────────────────────────────────────

class ModelRegisterRequest(BaseModel):
    """POST /v1/models/register 请求"""
    name: str = Field(..., description="模型名称")
    type: str = Field(default="classifier", description="模型类型: classifier / regressor")
    framework: str = Field(..., description="框架: xgboost / pytorch / lstm / tensorflow")
    version: str = Field(default="1.0.0", description="版本号")
    file_hash: str = Field(default="", description="模型文件SHA256哈希")


# ── 数据漂移 ────────────────────────────────────────────────────────────

class DriftCheckRequest(BaseModel):
    """POST /v1/data/drift-check 请求"""
    dataset_id: str = Field(..., description="数据集ID")
    reference_period: List[str] = Field(
        ..., description="基准期，如 ['2026-04-01', '2026-04-30']"
    )
    target_period: List[str] = Field(
        ..., description="目标期，如 ['2026-05-01', '2026-05-20']"
    )


# ── 报告生成 ────────────────────────────────────────────────────────────

class ReportGenerateRequest(BaseModel):
    """POST /v1/reports/generate 请求"""
    trace_ids: List[str] = Field(..., description="推理溯源ID列表")
    format: str = Field(default="html", description="输出格式: pdf / html")
    template: str = Field(default="compliance_v2", description="报告模板")
    include_visualizations: bool = Field(default=True, description="是否包含可视化")
