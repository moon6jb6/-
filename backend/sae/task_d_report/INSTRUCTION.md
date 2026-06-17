# 任务D：统一报告格式与鲁棒性测试框架

## 你的身份
你是智信引擎项目的质量工程师。你的任务是设计所有架构统一的归因报告格式，以及一个通用的鲁棒性测试框架。

## 背景
智信引擎支持5种架构（Transformer、CfC、LSTM、XGBoost、通用PyTorch），每种架构的归因方法不同，但最终报告格式必须统一——客户不需要知道你用的什么方法，只需要看到标准化的审计报告。

## 你需要做的事

### 1. 先读这些文件理解现有报告格式
- `C:\Users\26298\OneDrive\Desktop\全栈开发\backend\liquid_networks\shared\interpret_base.py` — 看ReportConfig和generate_explanation_report()
- `C:\Users\26298\OneDrive\Desktop\全栈开发\backend\sae\compliance\validate_compliance.py` — 看7项交叉验证的报告格式
- `C:\Users\26298\OneDrive\Desktop\全栈开发\backend\sae\compliance\causal_ablation.py` — 看causality_report_v1.0.json的结构
- `C:\Users\26298\OneDrive\Desktop\全栈开发\backend\sae\compliance\bias_detection.py` — 看bias_report.json的结构

### 2. 写 `report_schema.py`
统一报告JSON schema，所有架构的归因报告必须遵循此格式：
```python
UNIFIED_REPORT_SCHEMA = {
    "meta": {
        "report_id": "str, UUID",
        "timestamp": "str, ISO8601",
        "platform_version": "str, 如v0.1.0"
    },
    "model_info": {
        "architecture": "str, transformer|cfc|lstm|xgboost|generic_pytorch",
        "model_name": "str",
        "task": "str, 如二分类/多分类/回归",
        "input_shape": "list[int]",
        "output_shape": "list[int]"
    },
    "data_info": {
        "dataset_name": "str",
        "sample_count": "int",
        "feature_names": "list[str]",
        "data_hash": "str, SHA256（可选）"
    },
    "attribution": {
        "method": "str, rome|patching|ig|tree_shap|tau_analysis|...",
        "global_importance": {
            "feature_name": "float, 特征重要性值（按值排序）"
        },
        "single_sample": {
            "input": "list, 原始输入值",
            "output": "float, 模型输出概率",
            "attributions": "list[float], 每个特征的归因值",
            "top_features": [
                {"feature": "str", "attribution": "float", "rank": "int"}
            ]
        },
        "completeness_check": {
            "sum_attributions": "float",
            "output_minus_baseline": "float",
            "error_pct": "float, 应<1%"
        }
    },
    "robustness": {
        "consistency_std": "float, 3次运行标准差",
        "noise_top3_unchanged": "bool, 加噪声后top3是否不变",
        "ood_warning": "bool, 是否检测到OOD"
    },
    "conclusion": {
        "top3_features": "list[str], 最重要的3个特征",
        "confidence": "str, high|medium|low",
        "one_line_summary": "str, 一句话结论，中文"
    }
}

def validate_report(report: dict) -> tuple[bool, list[str]]:
    """验证报告是否符合schema，返回(是否合法, 错误列表)"""
    pass

def generate_empty_report() -> dict:
    """生成空报告模板"""
    pass

def convert_legacy_report(legacy_report: dict, source_type: str) -> dict:
    """
    将现有报告格式转换为统一格式

    Args:
        legacy_report: 现有报告（causality_report / bias_report / interpret报告等）
        source_type: "rome" | "bias" | "cfc_interpret" | "patching" | "drift"
    """
    pass
```

### 3. 写 `report_template.html`
面向非技术人员的HTML报告模板：
```html
<!-- 要求：
1. 不出现代码、不出现JSON、不出现技术术语
2. 中文
3. 响应式设计（手机也能看）
4. 内容结构：
   - 标题：AI模型决策审计报告
   - 模型名称和用途（一句话）
   - 数据来源说明
   - 核心发现：Top3最重要的特征（大字+颜色高亮）
   - 归因可视化：用HTML table + 背景色模拟热力图
   - 鲁棒性测试结果（✅/❌ 图标）
   - 一句话结论
   - 审计时间戳
5. 颜色方案：深蓝#1a365d标题，浅灰#f7fafc背景，绿色#38a169通过，红色#e53e3e未通过
-->
```

### 4. 写 `robustness_test.py`
鲁棒性测试框架：
```python
def test_attribution_consistency(model, input_tensor, attrib_fn, n_runs=3):
    """
    归因一致性测试：同一输入跑n_runs次，计算标准差
    通过标准：标准差<0.01
    返回：{"std": float, "pass": bool, "details": list}
    """
    pass

def test_boundary_conditions(model, attrib_fn, input_shape):
    """
    边界条件测试：全零输入、全一输入
    通过标准：不crash，归因值有合理解释
    返回：{"zero_input": {"crash": bool, "attribution_sum": float}, "ones_input": {...}}
    """
    pass

def test_adversarial_robustness(model, input_tensor, attrib_fn, noise_level=0.1):
    """
    对抗性测试：输入加noise_level比例的噪声，检查top3归因是否变化
    通过标准：top3归因排序不变
    返回：{"original_top3": list, "noisy_top3": list, "unchanged": bool}
    """
    pass

def test_ood_detection(model, input_tensor, attrib_fn):
    """
    OOD测试：用明显超出训练分布的数据做归因
    通过标准：能检测到OOD并给出警告
    返回：{"ood_detected": bool, "warning": str}
    """
    pass

def run_all_tests(model, input_tensor, attrib_fn, report_path=None):
    """
    运行全部鲁棒性测试，输出汇总报告
    """
    pass
```

### 5. 输出文件（全部放到本文件夹 `task_d_report\`）
- `report_schema.py` — 统一报告schema + 验证 + 转换
- `report_template.html` — HTML报告模板（中文，响应式）
- `robustness_test.py` — 鲁棒性测试框架
- `example_report.json` — 一份完整的示例报告（按schema填写的示例数据）
- `README.md` — 完成后写README，包含：
  - 完成状态（DONE/INCOMPLETE）
  - 产出文件清单
  - schema字段说明
  - HTML模板预览说明（浏览器打开即可查看）
  - 鲁棒性测试函数说明

## 验证标准
- validate_report() 能验证一个完整报告返回True
- validate_report() 对缺少必填字段的报告返回False
- HTML模板在浏览器打开能正常显示
- example_report.json 是合法JSON且通过validate_report
- robustness_test.py 的每个函数都有完整的返回值格式
- HTML不出现任何代码或JSON原文

## 注意
- 只在本文件夹内写文件
- 代码必须能独立运行
- HTML模板用纯CSS，不依赖外部库
- 鲁棒性测试框架的model和attrib_fn参数化，不绑定特定架构
- 完成后在README.md里标DONE
