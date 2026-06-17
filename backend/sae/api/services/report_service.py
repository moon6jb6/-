"""报告生成服务 — 使用 task_d_report 的 report_schema 和 report_template。"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

# 确保 SAE 根目录和 task_d_report 在路径中
SAE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SAE_ROOT not in sys.path:
    sys.path.insert(0, SAE_ROOT)

TASK_D_DIR = os.path.join(SAE_ROOT, "task_d_report")
if TASK_D_DIR not in sys.path:
    sys.path.insert(0, TASK_D_DIR)

from task_d_report.report_schema import generate_empty_report, validate_report

# 报告模板路径
TEMPLATE_PATH = os.path.join(SAE_ROOT, "task_d_report", "report_template.html")


def _load_template() -> str:
    """加载 HTML 报告模板。"""
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _render_template(template: str, report: dict[str, Any]) -> str:
    """用 report 数据渲染 HTML 模板。

    模板使用 {{key}} 占位符格式。
    """
    meta = report.get("meta", {})
    model_info = report.get("model_info", {})
    data_info = report.get("data_info", {})
    attribution = report.get("attribution", {})
    robustness = report.get("robustness", {})
    conclusion = report.get("conclusion", {})

    # Top 3 特征
    top3 = conclusion.get("top3_features", [])
    global_imp = attribution.get("global_importance", {})

    # 构建替换字典
    replacements = {
        "{{model_name}}": model_info.get("model_name", "N/A"),
        "{{task_description}}": model_info.get("task", "N/A"),
        "{{architecture}}": model_info.get("architecture", "N/A"),
        "{{input_shape}}": str(model_info.get("input_shape", "N/A")),
        "{{dataset_name}}": data_info.get("dataset_name", "N/A"),
        "{{sample_count}}": str(data_info.get("sample_count", 0)),
        "{{feature_count}}": str(len(data_info.get("feature_names", []))),
        "{{attribution_method}}": attribution.get("method", "N/A"),
        "{{timestamp}}": meta.get("timestamp", ""),
        "{{report_id}}": meta.get("report_id", ""),
        "{{one_line_summary}}": conclusion.get("one_line_summary", ""),
        "{{consistency_std}}": str(robustness.get("consistency_std", 0.0)),
        "{{completeness_error_pct}}": str(
            attribution.get("completeness_check", {}).get("error_pct", 0.0)
        ),
    }

    # Top 3 特征名和值
    for i in range(3):
        key_name = f"{{{{top{i+1}_name}}}}"
        key_val = f"{{{{top{i+1}_value}}}}"
        if i < len(top3):
            feat_name = top3[i]
            feat_val = global_imp.get(feat_name, 0.0)
            replacements[key_name] = feat_name
            replacements[key_val] = f"{feat_val:.4f}"
        else:
            replacements[key_name] = "-"
            replacements[key_val] = "0"

    # 替换置信度样式
    confidence = conclusion.get("confidence", "low")
    conf_map = {"high": "confidence-high", "medium": "confidence-medium", "low": "confidence-low"}
    conf_label_map = {"high": "置信度：高", "medium": "置信度：中", "low": "置信度：低"}

    # 替换模板
    html = template
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, str(value))

    # 替换鲁棒性测试结果的 badge
    consistency_pass = robustness.get("consistency_std", 999) < 0.01
    noise_pass = robustness.get("noise_top3_unchanged", True)
    ood_warning = robustness.get("ood_warning", False)

    # 替换置信度标签
    html = html.replace(
        '<span class="confidence confidence-high">置信度：高</span>',
        f'<span class="confidence {conf_map.get(confidence, "confidence-low")}">{conf_label_map.get(confidence, "置信度：低")}</span>'
    )

    # 替换鲁棒性 badges
    badge_pass = '<span class="badge badge-pass">&#10003;</span>'
    badge_fail = '<span class="badge badge-fail">&#10007;</span>'

    # 鲁棒性行替换（基于模板中的顺序：一致性、噪声、OOD）
    # 使用简单字符串替换，顺序匹配模板中的3个 <li>
    robustness_html = f'''<li>
                <span class="badge {"badge-pass" if consistency_pass else "badge-fail"}">{"&#10003;" if consistency_pass else "&#10007;"}</span>
                <span>归因一致性测试</span>
                <span class="robustness-detail">3次运行标准差: {robustness.get("consistency_std", 0.0)}</span>
            </li>
            <li>
                <span class="badge {"badge-pass" if noise_pass else "badge-fail"}">{"&#10003;" if noise_pass else "&#10007;"}</span>
                <span>噪声扰动测试</span>
                <span class="robustness-detail">{"加噪声后前三特征排序未变化" if noise_pass else "加噪声后前三特征排序发生变化"}</span>
            </li>
            <li>
                <span class="badge {"badge-fail" if ood_warning else "badge-pass"}">{"&#10007;" if ood_warning else "&#10003;"}</span>
                <span>分布外检测</span>
                <span class="robustness-detail">{"检测到超出训练分布的输入" if ood_warning else "输入在正常分布范围内"}</span>
            </li>'''

    # 替换原有鲁棒性列表
    import re
    html = re.sub(
        r'<ul class="robustness-list">.*?</ul>',
        f'<ul class="robustness-list">{robustness_html}</ul>',
        html,
        flags=re.DOTALL,
    )

    # 替换归因明细表格
    tbody_rows = ""
    sorted_features = sorted(global_imp.items(), key=lambda x: abs(x[1]), reverse=True)
    max_val = abs(sorted_features[0][1]) if sorted_features and sorted_features[0][1] != 0 else 1.0
    colors = ["#e53e3e", "#dd6b20", "#d69e2e", "#38a169", "#3182ce", "#805ad5"]
    for rank, (fname, fval) in enumerate(sorted_features, 1):
        pct = abs(fval) / max_val * 100 if max_val > 0 else 0
        color = colors[(rank - 1) % len(colors)]
        bar_chars = int(pct / 5)
        bar_str = "█" * max(bar_chars, 1)
        tbody_rows += f'''<tr>
                    <td>{rank}</td>
                    <td>{fname}</td>
                    <td>{fval:.4f}</td>
                    <td class="bar-cell">
                        <div class="bar-bg" style="width:{pct:.1f}%; background:{color};"></div>
                        <span class="bar-value">{bar_str}</span>
                    </td>
                </tr>\n'''

    html = re.sub(
        r'<tbody id="attribution-tbody">.*?</tbody>',
        f'<tbody id="attribution-tbody">{tbody_rows}</tbody>',
        html,
        flags=re.DOTALL,
    )

    return html


def generate_report(
    trace_ids: list[str],
    fmt: str = "html",
    template_name: str = "compliance_v2",
    include_visualizations: bool = True,
) -> dict[str, Any]:
    """生成统一报告。

    使用 report_schema.generate_empty_report() 创建基础报告，
    填充归因数据，然后用 report_template.html 渲染。
    """
    # 生成基础报告
    report = generate_empty_report()
    report["meta"]["report_id"] = str(uuid.uuid4())
    report["meta"]["timestamp"] = datetime.now(timezone.utc).isoformat()

    # 填充演示数据（生产环境中应从 trace_ids 查询实际归因结果）
    report["model_info"]["model_name"] = "智信引擎模型"
    report["model_info"]["task"] = "二分类"
    report["model_info"]["architecture"] = "xgboost"
    report["model_info"]["input_shape"] = [30]
    report["model_info"]["output_shape"] = [2]

    report["data_info"]["dataset_name"] = f"Trace集合 ({', '.join(trace_ids[:3])})"
    report["data_info"]["sample_count"] = len(trace_ids)
    report["data_info"]["feature_names"] = [
        "mean radius", "mean texture", "mean perimeter", "mean area",
        "mean smoothness", "mean compactness", "mean concavity",
    ]

    report["attribution"]["method"] = "tree_shap"
    report["attribution"]["global_importance"] = {
        "mean radius": 0.3421,
        "mean texture": 0.2156,
        "mean perimeter": 0.1834,
        "mean area": 0.1102,
        "mean smoothness": 0.0876,
        "mean compactness": 0.0611,
    }
    report["attribution"]["single_sample"]["top_features"] = [
        {"feature": "mean radius", "attribution": 0.3421, "rank": 1},
        {"feature": "mean texture", "attribution": 0.2156, "rank": 2},
        {"feature": "mean perimeter", "attribution": 0.1834, "rank": 3},
    ]
    report["attribution"]["completeness_check"]["error_pct"] = 0.0032

    report["robustness"]["consistency_std"] = 0.0023
    report["robustness"]["noise_top3_unchanged"] = True
    report["robustness"]["ood_warning"] = False

    report["conclusion"]["top3_features"] = ["mean radius", "mean texture", "mean perimeter"]
    report["conclusion"]["confidence"] = "high"
    report["conclusion"]["one_line_summary"] = (
        "模型决策主要依赖 mean radius、mean texture 和 mean perimeter 三个特征，"
        "归因一致性高，鲁棒性测试全部通过。"
    )

    # 验证报告格式
    is_valid, errors = validate_report(report)
    if not is_valid:
        raise ValueError(f"报告格式验证失败: {errors}")

    # 渲染 HTML
    template = _load_template()
    html_content = _render_template(template, report)

    report_id = report["meta"]["report_id"]
    now = datetime.now(timezone.utc).isoformat()

    return {
        "report_id": report_id,
        "format": fmt,
        "content": html_content,
        "trace_ids": trace_ids,
        "created_at": now,
    }
