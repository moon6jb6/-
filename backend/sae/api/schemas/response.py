"""Pydantic v2 响应模型 — 智信引擎 API"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── 通用错误 ────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    """错误详情"""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """统一错误响应"""
    error: ErrorDetail


# ── 推理溯源 ────────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    """预测结果"""
    label: str
    confidence: float


class ChainOfThoughtStep(BaseModel):
    """推理链步骤"""
    step: int
    action: str
    detail: str


class InferenceTraceResponse(BaseModel):
    """POST /v1/inference/trace 响应"""
    trace_id: str
    prediction: PredictionResult
    shap_values: Optional[Dict[str, Any]] = None
    lime_explanation: Optional[Dict[str, Any]] = None
    chain_of_thought: List[ChainOfThoughtStep] = []
    created_at: str


# ── 模型管理 ────────────────────────────────────────────────────────────

class ModelRegisterResponse(BaseModel):
    """POST /v1/models/register 响应"""
    model_id: str
    name: str
    framework: str
    version: str
    sha256_verified: bool = False
    created_at: str


class ModelVersion(BaseModel):
    """模型版本信息"""
    version: str
    file_hash: str
    sha256_verified: bool = False
    created_at: str


class ModelListResponse(BaseModel):
    """GET /v1/models 响应"""
    models: list
    total: int


class ModelDeleteResponse(BaseModel):
    """DELETE /v1/models/{model_id} 响应"""
    message: str


class ModelVersionsResponse(BaseModel):
    """GET /v1/models/{model_id}/versions 响应"""
    model_id: str
    name: str
    versions: List[ModelVersion]


# ── 数据漂移 ────────────────────────────────────────────────────────────

class DriftAlert(BaseModel):
    """漂移告警"""
    level: str = Field(description="告警级别: none / warning / critical")
    message: str


class DriftCheckResponse(BaseModel):
    """POST /v1/data/drift-check 响应"""
    dataset_id: str
    psi: float = Field(description="Population Stability Index")
    drift_level: str = Field(description="无漂移 / 轻微漂移 / 显著漂移")
    alert: DriftAlert
    feature_drift: Dict[str, float] = Field(description="各特征的PSI值")
    created_at: str


# ── 数据谱系 ────────────────────────────────────────────────────────────

class LineageNode(BaseModel):
    """谱系节点"""
    id: str
    name: str
    type: str = Field(description="节点类型: raw / transformed / derived")


class LineageEdge(BaseModel):
    """谱系边"""
    source: str
    target: str
    transformation: str = Field(description="转换操作描述")


class DataLineageResponse(BaseModel):
    """GET /v1/data/lineage/{dataset_id} 响应"""
    dataset_id: str
    parents: List[LineageNode] = []
    transformations: List[LineageEdge] = []
    children: List[LineageNode] = []
    created_at: str


# ── 报告生成 ────────────────────────────────────────────────────────────

class ReportGenerateResponse(BaseModel):
    """POST /v1/reports/generate 响应"""
    report_id: str
    format: str
    content: str = Field(description="报告内容（HTML字符串或PDF base64）")
    trace_ids: List[str]
    created_at: str
