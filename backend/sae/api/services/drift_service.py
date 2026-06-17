"""漂移检测服务 — 实现 PSI 计算。"""

from __future__ import annotations

import uuid
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np


def _distribution_from_csv(csv_path: str, columns: list = None,
                           bins: int = 10) -> dict:
    """从 CSV 文件计算特征分布。"""
    import pandas as pd
    df = pd.read_csv(csv_path)
    numeric_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    distributions = {}
    for col in numeric_cols:
        values = df[col].dropna().values
        if len(values) == 0:
            continue
        hist, _ = np.histogram(values, bins=bins)
        prob = hist.astype(float) + 1e-6
        prob = prob / prob.sum()
        distributions[col] = prob
    return distributions


def _generate_synthetic_distribution(seed: int, n_features: int = 5,
                                     n_bins: int = 10) -> dict:
    """生成合成数据分布（用于演示 PSI 计算）。"""
    rng = np.random.RandomState(seed)
    features = {}
    feature_names = ["income", "credit_score", "debt_ratio", "loan_amount", "employment_years"]
    for i in range(min(n_features, len(feature_names))):
        # 每个特征生成一个直方图分布
        raw = rng.randn(1000) * (10 + i * 5) + (50 + i * 20)
        hist, _ = np.histogram(raw, bins=n_bins)
        # 归一化为概率分布（加平滑避免零值）
        prob = hist.astype(float) + 1e-6
        prob = prob / prob.sum()
        features[feature_names[i]] = prob
    return features


def _psi_single(expected: np.ndarray, actual: np.ndarray) -> float:
    """计算单个特征的 PSI。

    PSI = Σ(actual% - expected%) * ln(actual% / expected%)
    """
    # 确保无零值
    expected = np.clip(expected, 1e-6, None)
    actual = np.clip(actual, 1e-6, None)

    # 归一化
    expected = expected / expected.sum()
    actual = actual / actual.sum()

    psi = np.sum((actual - expected) * np.log(actual / expected))
    return float(psi)


def _psi_to_drift_level(psi: float) -> str:
    """PSI 值转漂移级别。"""
    if psi < 0.1:
        return "无漂移"
    elif psi < 0.25:
        return "轻微漂移"
    else:
        return "显著漂移"


def _psi_to_alert(psi: float) -> dict[str, str]:
    """PSI 值转告警信息。"""
    if psi < 0.1:
        return {"level": "none", "message": "数据分布稳定，无显著漂移"}
    elif psi < 0.25:
        return {"level": "warning", "message": f"检测到轻微数据漂移 (PSI={psi:.4f})，建议关注"}
    else:
        return {"level": "critical", "message": f"检测到显著数据漂移 (PSI={psi:.4f})，需要排查原因"}


def compute_drift(
    dataset_id: str,
    reference_period: list,
    target_period: list,
    reference_file: str = None,
    target_file: str = None,
) -> dict[str, Any]:
    """计算数据漂移。支持从CSV文件加载真实分布。"""
    if reference_file and target_file:
        ref_dist = _distribution_from_csv(reference_file)
        tgt_dist = _distribution_from_csv(target_file)
    else:
        ref_seed = hash(f"{dataset_id}_{reference_period}") % (2**31)
        tgt_seed = hash(f"{dataset_id}_{target_period}") % (2**31)
        ref_dist = _generate_synthetic_distribution(seed=ref_seed)
        tgt_dist = _generate_synthetic_distribution(seed=tgt_seed)

    feature_drift: dict[str, float] = {}
    total_psi = 0.0
    n_features = 0

    for feature_name in ref_dist:
        if feature_name in tgt_dist:
            psi = _psi_single(ref_dist[feature_name], tgt_dist[feature_name])
            feature_drift[feature_name] = round(psi, 6)
            total_psi += psi
            n_features += 1

    avg_psi = total_psi / max(n_features, 1)

    now = datetime.now(timezone.utc).isoformat()

    return {
        "dataset_id": dataset_id,
        "psi": round(avg_psi, 6),
        "drift_level": _psi_to_drift_level(avg_psi),
        "alert": _psi_to_alert(avg_psi),
        "feature_drift": feature_drift,
        "created_at": now,
    }
