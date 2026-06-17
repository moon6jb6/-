# 任务A：XGBoost TreeSHAP推理归因

## 你的身份
你是智信引擎项目的AI可解释性工程师。你的任务是为XGBoost模型实现TreeSHAP推理归因，产出可运行的代码和标准化报告。

## 背景
智信引擎是一个AI决策可追溯平台，核心定位是"通过咱们的审核=通过法律的审核"（充分条件）。目前已覆盖Transformer和CfC两种架构的推理归因，现在需要补齐XGBoost。

## 你需要做的事

### 1. 先读这些文件理解现有框架
- `C:\Users\26298\OneDrive\Desktop\全栈开发\backend\liquid_networks\shared\interpret_base.py` — 看ReportConfig和报告生成的结构
- `C:\Users\26298\OneDrive\Desktop\全栈开发\backend\fraud_detect\shap_compare.py` — 看已有的SHAP集成代码（是GradientExplainer，你需要换成TreeExplainer）

### 2. 写 `tree_shap.py`
功能要求：
```python
import shap
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import json
import os

def train_xgboost_model():
    """用sklearn乳腺癌数据集训练XGBoost二分类模型"""
    # load_breast_cancer, 80/20划分, random_state=42
    # XGBClassifier, n_estimators=100
    # 返回 model, X_test, y_test, feature_names

def compute_treeSHAP(model, X_test, feature_names):
    """用TreeSHAP计算全局和单条归因"""
    # shap.TreeExplainer(model)
    # 全局：abs(shap_values).mean(axis=0) → 特征重要性排序
    # 单条：shap_values[0] → 第一条测试样本的归因
    # 完备性验证：shap_values[0].sum() + base_value ≈ model.predict_proba(X_test[[0]])[0,1]
    # 误差应<1%

def generate_shap_report(model, X_test, y_test, feature_names, output_dir):
    """生成JSON报告+HTML报告"""
    # JSON报告包含：
    # - model_type, dataset, task
    # - global_feature_importance（按重要性排序）
    # - single_sample_attribution（第一条样本）
    # - completeness_check（误差<1%）
    # - model_accuracy
    # HTML报告：简单表格，面向非技术人员，中文

if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    model, X_test, y_test, feature_names = train_xgboost_model()
    compute_treeSHAP(model, X_test, feature_names)
    generate_shap_report(model, X_test, y_test, feature_names, output_dir)
```

### 3. 输出文件（全部放到本文件夹 `task_a_xgboost\`）
- `tree_shap.py` — 可独立运行的完整代码
- `shap_report.json` — JSON格式归因报告
- `shap_report.html` — HTML格式报告（中文，不出现代码和JSON）
- `README.md` — 完成后写一个README，包含：
  - 完成状态（DONE/INCOMPLETE）
  - 产出文件清单
  - 运行方式（python tree_shap.py）
  - 验证结果（完备性误差、top特征是否符合预期）

## 验证标准
- SHAP值之和 ≈ 模型输出 - 基值（误差<1%）
- "mean radius" 和 "worst concavity" 应排在top5特征
- 代码能独立运行：`python tree_shap.py`
- JSON是合法JSON（json.dumps生成）
- HTML不出现代码，中文

## 注意
- 只在本文件夹内写文件，不要修改其他文件夹
- 代码必须能独立运行，不依赖其他Agent的产出
- 完成后在README.md里标DONE
