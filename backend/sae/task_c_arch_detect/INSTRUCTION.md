# 任务C：架构自动检测与归因路由

## 你的身份
你是智信引擎项目的系统架构工程师。你的任务是实现模型架构自动检测功能——用户上传任意模型，系统自动识别架构类型并路由到对应的归因方法。

## 背景
智信引擎需要支持5种架构的推理归因：Transformer、CfC/LNN、LSTM/GRU、XGBoost、通用PyTorch。目前每种架构的归因方法各自独立实现，需要一个统一的检测和路由层。

## 你需要做的事

### 1. 先读这些文件了解各架构的代码特征
- `C:\Users\26298\OneDrive\Desktop\全栈开发\backend\sae\compliance\causal_ablation.py` — Transformer用GPTNeoXForCausalLM
- `C:\Users\26298\OneDrive\Desktop\全栈开发\backend\sae\compliance\activation_patching.py` — Transformer用GPTNeoXForCausalLM
- `C:\Users\26298\OneDrive\Desktop\全栈开发\backend\liquid_networks\shared\interpret_base.py` — CfC用ncps.torch.CfC
- `C:\Users\26298\OneDrive\Desktop\全栈开发\backend\flash_crash\interpret.py` — CfC模型结构
- `C:\Users\26298\OneDrive\Desktop\全栈开发\backend\fraud_detect\shap_compare.py` — SHAP相关import

### 2. 写 `arch_detect.py`
功能要求：
```python
import torch.nn as nn

# 需要识别的5种架构及其判断逻辑
ARCHITECTURE_REGISTRY = {
    "cfc": {
        "priority": 1,  # 最高优先级，因为CfC也是nn.Module
        "check": lambda model: (
            type(model).__name__ == 'CfC' or
            hasattr(model, 'ncps') or
            'liquid' in type(model).__module__.lower() if hasattr(type(model), '__module__') else False
        ),
        "explain_method": "tau_ig_trajectory",
        "description": "CfC/LNN液态神经网络"
    },
    "transformer": {
        "priority": 2,
        "check": lambda model: (
            'GPTNeoX' in type(model).__name__ or
            'GPT2' in type(model).__name__ or
            'Bert' in type(model).__name__ or
            hasattr(model, 'transformer') and hasattr(model, 'lm_head') or
            'transformer' in type(model).__module__.lower() if hasattr(type(model), '__module__') else False
        ),
        "explain_method": "rome_and_patching",
        "description": "Transformer架构（GPT/BERT等）"
    },
    "lstm": {
        "priority": 3,
        "check": lambda model: (
            isinstance(model, (nn.LSTM, nn.GRU)) or
            (hasattr(model, 'lstm') and isinstance(model.lstm, (nn.LSTM, nn.GRU))) or
            (hasattr(model, 'rnn') and isinstance(model.rnn, (nn.LSTM, nn.GRU)))
        ),
        "explain_method": "integrated_gradients",
        "description": "LSTM/GRU循环神经网络"
    },
    "xgboost": {
        "priority": 4,
        "check": lambda model: (
            type(model).__module__.startswith('xgboost') if hasattr(type(model), '__module__') else False
        ),
        "explain_method": "tree_shap",
        "description": "XGBoost梯度提升树"
    },
    "generic_pytorch": {
        "priority": 99,  # 最低优先级，兜底
        "check": lambda model: isinstance(model, nn.Module),
        "explain_method": "integrated_gradients",
        "description": "通用PyTorch模型"
    }
}

def detect_architecture(model) -> dict:
    """
    检测模型架构类型

    Args:
        model: 任意模型对象

    Returns:
        {
            "architecture": "lstm",
            "explain_method": "integrated_gradients",
            "description": "LSTM/GRU循环神经网络",
            "confidence": "high"
        }

    Raises:
        ValueError: 无法识别的架构
    """
    # 按priority从低到高检查（priority越小越优先）
    # generic_pytorch是兜底，永远能匹配nn.Module
    pass

def get_supported_architectures() -> list:
    """返回所有支持的架构列表"""
    pass

def validate_model_input(model, sample_input) -> bool:
    """验证模型能接受给定的输入格式"""
    pass
```

### 3. 写 `detect_and_explain.py`
功能要求：
```python
def detect_and_explain(model, input_data, **kwargs) -> dict:
    """
    自动检测架构并运行对应归因

    流程：
    1. detect_architecture(model) → 获取架构类型
    2. 根据架构类型路由到对应归因方法
    3. 返回归因结果

    Returns:
        {
            "architecture": "lstm",
            "attribution_method": "integrated_gradients",
            "attributions": tensor_or_array,
            "report": {...}
        }
    """
    # 路由逻辑：
    # cfc → 调用 interpret_base.py 的方法（只写接口，不实际调用，因为依赖其他Agent的产出）
    # transformer → 调用 sae.compliance 的方法（只写接口）
    # lstm → 调用 ig_generic.py（只写接口）
    # xgboost → 调用 tree_shap.py（只写接口）
    # generic_pytorch → 调用 ig_generic.py（只写接口）
    #
    # 注意：这里只写路由框架，实际归因函数由Agent A和B实现
    # 你可以写mock函数来测试路由逻辑
    pass
```

### 4. 写 `test_arch_detect.py`
单元测试，覆盖：
```python
# 测试用例：
# 1. nn.LSTM → "lstm"
# 2. nn.GRU → "lstm"（GRU归入LSTM类）
# 3. 包含lstm属性的自定义模型 → "lstm"
# 4. nn.Module（MLP）→ "generic_pytorch"
# 5. xgboost.XGBClassifier → "xgboost"
# 6. CfC模型（如果能构造mock）→ "cfc"
# 7. 未知非nn.Module类型 → ValueError
# 8. get_supported_architectures() 返回5种架构
# 9. validate_model_input 对正确/错误输入的处理
```

### 5. 输出文件（全部放到本文件夹 `task_c_arch_detect\`）
- `arch_detect.py` — 架构检测核心模块
- `detect_and_explain.py` — 检测+归因路由
- `test_arch_detect.py` — 单元测试（可独立运行）
- `README.md` — 完成后写README，包含：
  - 完成状态（DONE/INCOMPLETE）
  - 产出文件清单
  - 运行方式（python test_arch_detect.py）
  - 测试结果（多少用例通过）
  - 每种架构的判断逻辑说明

## 验证标准
- 5种架构全部能正确识别
- 单元测试全部通过
- 优先级正确：CfC > Transformer > LSTM > XGBoost > Generic PyTorch
- generic_pytorch是兜底，任何nn.Module都能匹配
- xgboost检测不依赖PyTorch

## 注意
- 只在本文件夹内写文件
- 代码必须能独立运行
- 对于Agent A和B的归因函数，用mock/placeholder代替，重点测路由逻辑
- 完成后在README.md里标DONE
