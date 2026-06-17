"""漂移检测服务 — 实现 PSI 计算。"""

from __future__ import annotations

import uuid
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np


def _generate_synthetic_distribution(seed: int, n_features: int = 5,
                                     n_bins: int = 10) -> dict[str, np.ndarray]:
    """生成合成数据分布（用于演示 PSI 计算）。

    实际生产中应从数据集加载真实分布。
    """
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
    reference_period: list[str],
    target_period: list[str],
) -> dict[str, Any]:
    """计算数据漂移。

    实现真实的 PSI 计算逻辑。
    当前使用合成数据演示；生产环境中应连接实际数据源。
    """
    # 用数据集ID和时间段作为种子，确保同一参数返回一致结果
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
