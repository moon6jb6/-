# 任务A：XGBoost TreeSHAP 推理归因

## 完成状态

**DONE**

## 产出文件清单

| 文件 | 说明 |
|------|------|
| `tree_shap.py` | 可独立运行的完整代码（XGBoost训练 + TreeSHAP归因 + 报告生成） |
| `shap_report.json` | JSON格式归因报告（json.dumps生成，合法JSON） |
| `shap_report.html` | HTML格式报告（中文，不出现代码和JSON） |
| `README.md` | 本文件 |

## 运行方式

```bash
python tree_shap.py
```

## 验证结果

### 1. SHAP 完备性验证

| 项目 | 数值 |
|------|------|
| 基值 (base_value) | 0.668790 |
| SHAP 值之和 | 5.226017 |
| 基值 + SHAP 和 | 5.894807 |
| 模型 margin 输出 | 5.894808 |
| 绝对误差 | 1.31e-06 |
| 相对误差 | 0.0000% |
| **结果** | **PASS (< 1%)** |

### 2. 模型准确率

- 测试集准确率: **95.61%**
- 数据集: sklearn breast_cancer (569样本, 30特征)
- 划分: 80/20, random_state=42

### 3. 全局特征重要性 Top 10

| 排名 | 特征 | 平均|SHAP| |
|------|------|-----------|
| 1 | mean concave points | 1.251806 |
| 2 | worst area | 1.177029 |
| 3 | worst concave points | 0.975928 |
| 4 | worst texture | 0.846842 |
| 5 | area error | 0.787382 |
| 6 | worst concavity | 0.782375 |
| 7 | compactness error | 0.588523 |
| 8 | worst symmetry | 0.420688 |
| 9 | mean texture | 0.419982 |
| 10 | symmetry error | 0.364100 |

### 4. 关于 top5 特征预期的说明

INSTRUCTION 中期望 "mean radius" 和 "worst concavity" 进入 top5。实际结果：

- **worst concavity** 排名第 6（0.782375），与第 5 的 area error（0.787382）仅差 0.6%，非常接近。
- **mean radius** 排名第 23（0.034502），因 XGBoost 将其重要性分散到高度相关的 "mean concave points" 和 "worst radius" 等特征上，导致单独的 "mean radius" SHAP 值较低。这是树模型处理多重共线性特征的典型行为。
- Top 5 特征（mean concave points, worst area, worst concave points, worst texture, area error）均为医学上公认的乳腺癌关键诊断指标，归因结果合理。

### 5. 其他验证

- [x] JSON 合法（json.dumps 生成）
- [x] HTML 中文，不出现代码和 JSON
- [x] 代码可独立运行（`python tree_shap.py`）
