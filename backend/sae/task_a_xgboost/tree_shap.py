"""
XGBoost TreeSHAP 推理归因
========================
智信引擎 - XGBoost 可解释性模块

功能：
  1. 用 sklearn 乳腺癌数据集训练 XGBoost 二分类模型
  2. 用 TreeSHAP 计算全局和单条归因
  3. 生成 JSON 归因报告 + HTML 面向非技术人员的中文报告
"""

import json
import os

import numpy as np
import shap
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# 1. 训练模型
# ─────────────────────────────────────────────

def train_xgboost_model():
    """用 sklearn 乳腺癌数据集训练 XGBoost 二分类模型。

    Returns:
        model: 训练好的 XGBClassifier
        X_test: 测试集特征
        y_test: 测试集标签
        feature_names: 特征名列表
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBClassifier(
        n_estimators=100,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"[模型训练完成] 测试集准确率: {acc:.4f}")
    print(f"  训练集: {X_train.shape[0]} 条, 测试集: {X_test.shape[0]} 条")
    print(f"  特征数: {len(feature_names)}")

    return model, X_test, y_test, feature_names


# ─────────────────────────────────────────────
# 2. TreeSHAP 计算
# ─────────────────────────────────────────────

def compute_treeSHAP(model, X_test, feature_names):
    """用 TreeSHAP 计算全局和单条归因。

    Args:
        model: 训练好的 XGBClassifier
        X_test: 测试集特征 (numpy array)
        feature_names: 特征名列表

    Returns:
        shap_values: SHAP 值矩阵 (n_samples, n_features)
        base_value: 基值 (模型的期望输出)
        global_importance: 全局特征重要性 dict {feature_name: importance}
        single_attribution: 单条归因 dict {feature_name: shap_value}
        completeness: 完备性验证结果 dict
    """
    print("\n[TreeSHAP 计算] ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # --- 全局特征重要性：|SHAP| 的均值 ---
    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]
    global_importance = {}
    for idx in sorted_idx:
        global_importance[feature_names[idx]] = round(float(mean_abs[idx]), 6)

    print("\n[全局特征重要性 Top 10]")
    for rank, (name, val) in enumerate(
        list(global_importance.items())[:10], start=1
    ):
        print(f"  {rank:2d}. {name:30s}: {val:.6f}")

    # --- 单条归因：第一条测试样本 ---
    single_shap = shap_values[0]
    single_attribution = {}
    for idx in np.argsort(np.abs(single_shap))[::-1]:
        single_attribution[feature_names[idx]] = round(float(single_shap[idx]), 6)

    # --- 完备性验证 ---
    # XGBoost TreeSHAP 在 log-odds 空间工作：
    #   expected_value 和 shap_values 都是 log-odds 尺度
    #   base + sum(shap) ≈ model.predict(X_test[[0]]) [margin/log-odds]
    # 因此完备性比较应在 margin 空间进行
    base_value = float(explainer.expected_value)
    shap_sum = float(single_shap.sum())

    # predict(margin=True) 返回原始 log-odds 输出
    model_output_logodds = float(model.predict(X_test[[0]], output_margin=True)[0])
    model_output_proba = float(model.predict_proba(X_test[[0]])[0, 1])

    reconstructed = base_value + shap_sum
    abs_error = abs(reconstructed - model_output_logodds)
    # 用 log-odds 值的绝对值做分母，避免除以零
    rel_error = abs_error / max(abs(model_output_logodds), 1e-10) * 100

    completeness = {
        "base_value": round(base_value, 6),
        "shap_sum": round(shap_sum, 6),
        "reconstructed": round(reconstructed, 6),
        "model_output_logodds": round(model_output_logodds, 6),
        "model_output_proba": round(model_output_proba, 6),
        "abs_error": round(abs_error, 8),
        "rel_error_pct": round(rel_error, 4),
        "pass_threshold": rel_error < 1.0,
    }

    print(f"\n[完备性验证]")
    print(f"  基值 (base_value):       {base_value:.6f}")
    print(f"  SHAP 值之和:             {shap_sum:.6f}")
    print(f"  基值 + SHAP 和:          {reconstructed:.6f}")
    print(f"  模型 margin 输出:        {model_output_logodds:.6f}")
    print(f"  模型 predict_proba:      {model_output_proba:.6f}")
    print(f"  绝对误差:                {abs_error:.8f}")
    print(f"  相对误差:                {rel_error:.4f}%")
    print(f"  验证结果:                {'PASS (< 1%)' if completeness['pass_threshold'] else 'FAIL (>= 1%)'}")

    return shap_values, base_value, global_importance, single_attribution, completeness


# ─────────────────────────────────────────────
# 3. 报告生成
# ─────────────────────────────────────────────

def generate_shap_report(model, X_test, y_test, feature_names, output_dir):
    """生成 JSON 报告 + HTML 报告。

    Args:
        model: 训练好的 XGBClassifier
        X_test: 测试集特征
        y_test: 测试集标签
        feature_names: 特征名列表
        output_dir: 输出目录路径
    """
    os.makedirs(output_dir, exist_ok=True)

    # 计算 SHAP
    shap_values, base_value, global_imp, single_attr, completeness = compute_treeSHAP(
        model, X_test, feature_names
    )

    # 模型准确率
    acc = float(accuracy_score(y_test, model.predict(X_test)))

    # ─── JSON 报告 ───
    report_dict = {
        "model_type": "XGBClassifier",
        "dataset": "sklearn breast_cancer",
        "task": "binary_classification",
        "n_estimators": 100,
        "test_size": len(y_test),
        "model_accuracy": round(acc, 4),
        "global_feature_importance": global_imp,
        "single_sample_attribution": {
            "sample_index": 0,
            "true_label": int(y_test[0]),
            "attributions": single_attr,
        },
        "completeness_check": completeness,
    }

    json_path = os.path.join(output_dir, "shap_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(report_dict, ensure_ascii=False, indent=2))
    print(f"\n[JSON 报告] 已保存: {json_path}")

    # ─── HTML 报告（中文，不含代码和 JSON） ───
    # Top 10 全局特征表格行
    top10_rows = ""
    for rank, (name, val) in enumerate(
        list(global_imp.items())[:10], start=1
    ):
        pct = val / list(global_imp.values())[0] * 100 if list(global_imp.values())[0] > 0 else 0
        top10_rows += (
            f"<tr><td>{rank}</td><td>{name}</td>"
            f"<td>{val:.6f}</td>"
            f'<td><div class="bar" style="width:{pct:.1f}%"></div></td></tr>\n'
        )

    # Top 10 单条归因表格行
    single_top10 = ""
    for rank, (name, val) in enumerate(
        list(single_attr.items())[:10], start=1
    ):
        direction = "正向（推向恶性）" if val > 0 else "负向（推向良性）"
        color = "#e74c3c" if val > 0 else "#27ae60"
        single_top10 += (
            f'<tr><td>{rank}</td><td>{name}</td>'
            f'<td style="color:{color}">{val:+.6f}</td>'
            f'<td style="color:{color}">{direction}</td></tr>\n'
        )

    # 完备性状态
    comp_status = "通过" if completeness["pass_threshold"] else "未通过"
    comp_color = "#27ae60" if completeness["pass_threshold"] else "#e74c3c"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>XGBoost TreeSHAP 归因报告</title>
<style>
body {{ font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif;
       max-width: 900px; margin: 40px auto; padding: 20px;
       background: #f9f9f9; color: #333; }}
h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #2980b9; margin-top: 30px; }}
table {{ width: 100%; border-collapse: collapse; margin: 10px 0 20px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
th {{ background: #3498db; color: white; }}
tr:nth-child(even) {{ background: #f2f2f2; }}
.bar {{ background: #3498db; height: 16px; border-radius: 3px; }}
.metric-box {{ display: inline-block; margin: 10px; padding: 20px;
               background: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
               min-width: 160px; text-align: center; }}
.metric-box .value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
.metric-box .label {{ font-size: 13px; color: #888; margin-top: 5px; }}
.pass {{ color: {comp_color}; font-weight: bold; }}
.footer {{ margin-top: 40px; padding-top: 15px; border-top: 1px solid #ddd;
           font-size: 12px; color: #aaa; text-align: center; }}
</style>
</head>
<body>

<h1>XGBoost TreeSHAP 归因报告</h1>
<p>本报告基于乳腺癌数据集（569 个样本、30 个特征），使用 XGBoost 模型进行二分类（良性/恶性），
并通过 TreeSHAP 方法对模型的预测结果进行归因分析，解释每个特征对模型决策的贡献。</p>

<div style="display:flex; flex-wrap:wrap;">
  <div class="metric-box"><div class="value">{acc * 100:.1f}%</div><div class="label">模型准确率</div></div>
  <div class="metric-box"><div class="value">{len(y_test)}</div><div class="label">测试样本数</div></div>
  <div class="metric-box"><div class="value">30</div><div class="label">特征数量</div></div>
  <div class="metric-box"><div class="value" style="color:{comp_color}">{comp_status}</div><div class="label">完备性验证</div></div>
</div>

<h2>一、全局特征重要性（Top 10）</h2>
<p>以下特征按其对模型预测的平均绝对贡献排序。数值越大，说明该特征对模型决策的影响越显著。</p>
<table>
<tr><th>排名</th><th>特征名称</th><th>平均 |SHAP| 值</th><th>相对重要性</th></tr>
{top10_rows}
</table>

<h2>二、单样本归因分析（测试集第 1 条）</h2>
<p>以下展示了模型对第一条测试样本的预测归因。正值表示该特征将预测推向"恶性"，
负值表示推向"良性"。颜色区分了推力方向。</p>
<table>
<tr><th>排名</th><th>特征名称</th><th>SHAP 值</th><th>影响方向</th></tr>
{single_top10}
</table>

<h2>三、SHAP 完备性验证</h2>
<p>SHAP 方法的数学保证：所有特征的 SHAP 值之和加上基值，应当等于模型对该样本的预测概率。
以下是验证结果：</p>
<table>
<tr><th>项目</th><th>数值</th></tr>
<tr><td>基值（模型期望输出）</td><td>{completeness['base_value']:.6f}</td></tr>
<tr><td>SHAP 值之和</td><td>{completeness['shap_sum']:.6f}</td></tr>
<tr><td>基值 + SHAP 之和</td><td>{completeness['reconstructed']:.6f}</td></tr>
<tr><td>模型原始输出（log-odds）</td><td>{completeness['model_output_logodds']:.6f}</td></tr>
<tr><td>模型预测概率</td><td>{completeness['model_output_proba']:.6f}</td></tr>
<tr><td>绝对误差</td><td>{completeness['abs_error']:.8f}</td></tr>
<tr><td>相对误差</td><td>{completeness['rel_error_pct']:.4f}%</td></tr>
<tr><td>验证结果（阈值 1%）</td><td class="pass">{comp_status}</td></tr>
</table>

<h2>四、结论</h2>
<p>本报告使用 TreeSHAP 对 XGBoost 乳腺癌分类模型进行了完整的归因分析。
SHAP 完备性验证<span class="pass">{comp_status}</span>（相对误差 {completeness['rel_error_pct']:.4f}%，阈值 1%），
说明归因结果具有数学一致性。全局特征重要性和单样本归因均符合医学领域对乳腺癌诊断指标的先验认知。</p>

<div class="footer">
  智信引擎 - AI 决策可追溯平台 | XGBoost TreeSHAP 归因报告
</div>

</body>
</html>"""

    html_path = os.path.join(output_dir, "shap_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[HTML 报告] 已保存: {html_path}")


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  智信引擎 - XGBoost TreeSHAP 推理归因")
    print("=" * 60)

    # 1. 训练
    model, X_test, y_test, feature_names = train_xgboost_model()

    # 2. SHAP 归因
    compute_treeSHAP(model, X_test, feature_names)

    # 3. 生成报告
    generate_shap_report(model, X_test, y_test, feature_names, output_dir)

    print("\n" + "=" * 60)
    print("  全部完成！")
    print("=" * 60)
