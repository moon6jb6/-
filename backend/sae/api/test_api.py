"""API 端点测试脚本。

用法（先启动服务器）:
    python api/run.py &
    python api/test_api.py
"""

import sys
import os
import json

# 确保 SAE 根目录在路径中
SAE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SAE_ROOT not in sys.path:
    sys.path.insert(0, SAE_ROOT)

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    """测试健康检查"""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    print("[PASS] GET /health")


def test_register_model():
    """测试模型注册"""
    resp = client.post("/v1/models/register", json={
        "name": "Credit Scoring v2",
        "type": "classifier",
        "framework": "xgboost",
        "version": "2.3.1",
        "file_hash": "sha256:a3f28b",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "model_id" in data
    assert data["model_id"].startswith("mdl_")
    assert data["name"] == "Credit Scoring v2"
    print("[PASS] POST /v1/models/register -> model_id:", data["model_id"])
    return data["model_id"]


def test_model_versions(model_id):
    """测试模型版本查询"""
    resp = client.get("/v1/models/{}/versions".format(model_id))
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == model_id
    assert len(data["versions"]) >= 1
    print("[PASS] GET /v1/models/{}/versions".format(model_id))


def test_inference_trace(model_id):
    """测试推理溯源"""
    resp = client.post("/v1/inference/trace", json={
        "model_id": model_id,
        "input_data": {"features": {"income": 75000, "credit_score": 720, "debt_ratio": 0.38}},
        "methods": ["shap"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "trace_id" in data
    assert data["trace_id"].startswith("trc_")
    assert "prediction" in data
    assert "shap_values" in data
    assert "chain_of_thought" in data
    print("[PASS] POST /v1/inference/trace -> trace_id:", data["trace_id"])
    print("       Prediction:", data["prediction"])
    return data["trace_id"]


def test_drift_check():
    """测试漂移检测"""
    resp = client.post("/v1/data/drift-check", json={
        "dataset_id": "ds_test_001",
        "reference_period": ["2026-04-01", "2026-04-30"],
        "target_period": ["2026-05-01", "2026-05-20"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "psi" in data
    assert "drift_level" in data
    assert "alert" in data
    assert "feature_drift" in data
    assert isinstance(data["psi"], float)
    assert data["psi"] >= 0
    print("[PASS] POST /v1/data/drift-check -> PSI:", data["psi"], "Level:", data["drift_level"])


def test_data_lineage():
    """测试数据谱系"""
    resp = client.get("/v1/data/lineage/ds_test_001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["dataset_id"] == "ds_test_001"
    assert "parents" in data
    assert "transformations" in data
    assert "children" in data
    print("[PASS] GET /v1/data/lineage/ds_test_001 -> parents:", len(data["parents"]), "children:", len(data["children"]))


def test_report_generate():
    """测试报告生成"""
    resp = client.post("/v1/reports/generate", json={
        "trace_ids": ["trc_test1", "trc_test2"],
        "format": "html",
        "template": "compliance_v2",
        "include_visualizations": True,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "report_id" in data
    assert data["format"] == "html"
    assert len(data["content"]) > 1000  # HTML content should be substantial
    assert "trc_test1" in data["trace_ids"]
    print("[PASS] POST /v1/reports/generate -> report_id:", data["report_id"], "content_length:", len(data["content"]))


def test_error_handling():
    """测试错误处理"""
    # 查询不存在的模型
    resp = client.get("/v1/models/mdl_nonexistent/versions")
    assert resp.status_code == 404
    print("[PASS] GET /v1/models/mdl_nonexistent/versions -> 404")

    # 对不存在的模型做推理
    resp = client.post("/v1/inference/trace", json={
        "model_id": "mdl_nonexistent",
        "input_data": {"features": {"x": 1}},
        "methods": ["shap"],
    })
    assert resp.status_code == 404
    print("[PASS] POST /v1/inference/trace (invalid model) -> 404")


if __name__ == "__main__":
    print("=" * 50)
    print("  智信引擎 API 端点测试")
    print("=" * 50)
    print()

    test_health()
    model_id = test_register_model()
    test_model_versions(model_id)
    trace_id = test_inference_trace(model_id)
    test_drift_check()
    test_data_lineage()
    test_report_generate()
    test_error_handling()

    print()
    print("=" * 50)
    print("  全部测试通过！")
    print("=" * 50)
