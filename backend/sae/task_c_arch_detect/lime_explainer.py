"""LIME 归因解释器 — 基于 sklearn 的局部线性代理模型实现。

不依赖 lime 库，使用扰动采样 + Ridge 回归实现等效的 LIME 算法。
"""
from __future__ import annotations

import numpy as np
from typing import Any, Callable, Dict, List


def explain_with_lime(
    model_predict_fn: Callable[[np.ndarray], np.ndarray],
    training_data: np.ndarray,
    sample: np.ndarray,
    feature_names: List[str],
    num_features: int = 10,
    n_samples: int = 500,
) -> Dict[str, Any]:
    """使用局部线性代理模型对单个样本做归因解释（LIME 等效实现）。

    Args:
        model_predict_fn: 接受 2D array (n_samples, n_features)，返回概率 (n_samples, n_classes)
        training_data: 训练数据 numpy array，用于估计特征分布
        sample: 要解释的单个样本 (1D array)
        feature_names: 特征名列表
        num_features: 展示前 N 个特征
        n_samples: 扰动采样数量

    Returns:
        dict with 'feature_attributions', 'intercept', 'prediction', 'score', 'method'
    """
    from sklearn.linear_model import Ridge

    n_features = len(sample)

    # 1. 以 sample 为中心扰动采样
    mean = training_data.mean(axis=0)
    std = training_data.std(axis=0) + 1e-8
    noise = np.random.randn(n_samples, n_features) * std * 0.1
    perturbed = sample + noise

    # 2. 获取扰动样本的预测
    perturbed_all = np.vstack([sample.reshape(1, -1), perturbed])
    predictions = model_predict_fn(perturbed_all)

    # 取正类概率
    if predictions.ndim == 2:
        pred_positive = predictions[:, 1]
    else:
        pred_positive = predictions

    sample_pred = float(pred_positive[0])
    perturbed_pred = pred_positive[1:]

    # 3. 计算距离权重（指数核）
    distances = np.sqrt(np.sum((perturbed - sample) ** 2, axis=1))
    bandwidth = np.std(distances) + 1e-8
    weights = np.exp(-(distances ** 2) / (bandwidth ** 2))

    # 4. 标准化扰动数据
    perturbed_normalized = (perturbed - mean) / std

    # 5. 拟合加权 Ridge 回归
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(perturbed_normalized, perturbed_pred, sample_weight=weights)

    # 6. 提取归因
    coefs = model.coef_[:n_features]
    intercept = float(model.intercept_)

    # 按绝对值排序
    feature_attributions = {}
    for i in range(min(n_features, num_features)):
        if i < len(feature_names):
            feature_attributions[feature_names[i]] = round(float(coefs[i]), 6)

    # R² score
    score = float(model.score(perturbed_normalized, perturbed_pred, sample_weight=weights))

    return {
        "feature_attributions": feature_attributions,
        "intercept": round(intercept, 6),
        "prediction": round(sample_pred, 6),
        "score": round(score, 6),
        "method": "lime",
        "n_samples": n_samples,
    }
