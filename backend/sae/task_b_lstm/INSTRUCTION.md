# 任务B：LSTM模型训练 + 通用Integrated Gradients实现

## 你的身份
你是智信引擎项目的AI可解释性工程师。你的任务是训练一个LSTM股票预测模型，并实现通用的Integrated Gradients归因方法（适用于任意PyTorch模型）。

## 背景
智信引擎是一个AI决策可追溯平台，核心定位是"通过咱们的审核=通过法律的审核"。目前已覆盖Transformer和CfC的推理归因，现在需要补齐LSTM/GRU和通用PyTorch。你做的通用IG框架将同时服务于LSTM和通用PyTorch两种架构。

## 你需要做的事

### 1. 先读这些文件理解现有框架
- `C:\Users\26298\OneDrive\Desktop\全栈开发\backend\flash_crash\interpret.py` L120-151 — 看现有CfC专用IG实现，你需要泛化它
- `C:\Users\26298\OneDrive\Desktop\全栈开发\backend\liquid_networks\shared\interpret_base.py` — 看ReportConfig和报告生成结构

### 2. 写 `lstm_model.py`
功能要求：
```python
import torch
import torch.nn as nn
import numpy as np

class StockLSTM(nn.Module):
    """LSTM股票方向预测模型"""
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)  # 涨/跌二分类

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后时间步
        return out

def prepare_stock_data():
    """准备股票数据"""
    # 优先尝试用akshare获取5分钟K线数据
    # 如果akshare不可用，用随机时序数据模拟（特征：open,high,low,close,volume）
    # seq_len=30, 标签：下一时间步涨(1)跌(0)
    # 返回 X_train, X_test, y_train, y_test, feature_names

def train_lstm(model, X_train, y_train, epochs=50):
    """训练LSTM模型"""
    # CrossEntropyLoss, Adam, lr=0.001
    # 返回训练好的model和accuracy

def save_checkpoint(model, path):
    """保存模型checkpoint"""
    torch.save(model.state_dict(), path)
```

### 3. 写 `ig_generic.py`
功能要求：
```python
def integrated_gradients(model, input_tensor, baseline=None, n_steps=50, target_class=1):
    """
    通用Integrated Gradients，适用于任意PyTorch模型

    关键泛化点（对比interpret.py L120-151）：
    - 原代码 prob = torch.sigmoid(logits[:, -1, :]) 硬编码了CfC输出格式
    - 你需要处理：序列输出（取最后时间步）vs 单步输出
    - 原代码 logits = model(inp) 假设单一tensor输入
    - 你需要处理LSTM可能需要的 (input, (h0, c0)) 元组输入

    Args:
        model: PyTorch模型
        input_tensor: 输入tensor (batch, seq_len, features) 或 (batch, features)
        baseline: 基线输入，默认全零
        n_steps: 插值步数
        target_class: 目标类别

    Returns:
        attributions: 归因矩阵，与input_tensor同形状
        completeness_error: 完备性误差（应<1%）
    """
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    scaled_inputs = [baseline + (float(i)/n_steps) * (input_tensor - baseline) for i in range(n_steps+1)]

    grads = []
    for x in scaled_inputs:
        x = x.requires_grad_(True)
        output = model(x)
        # 处理不同输出格式
        if output.dim() > 1 and output.shape[1] > 1:
            target_output = output[:, target_class]
        else:
            target_output = output.squeeze()
        target_output.backward()
        grads.append(x.grad.detach())

    avg_grads = torch.stack(grads).mean(dim=0)
    attributions = (input_tensor - baseline) * avg_grads

    # 完备性验证
    with torch.no_grad():
        output_final = model(input_tensor)
        baseline_output = model(baseline)
        if output_final.dim() > 1:
            expected = output_final[:, target_class] - baseline_output[:, target_class]
        else:
            expected = output_final - baseline_output
        actual = attributions.sum()
        completeness_error = abs(actual - expected) / (abs(expected) + 1e-8)

    return attributions, completeness_error.item()
```

### 4. 写 `generate_ig_report.py`
功能要求：
- 用训练好的LSTM模型，取一条测试样本
- 调用ig_generic.py计算归因
- 生成JSON报告+HTML报告
- JSON报告格式：
```json
{
  "model_type": "LSTM",
  "task": "stock_direction_prediction",
  "input_shape": [30, 5],
  "features": ["open", "high", "low", "close", "volume"],
  "single_sample_attribution": {
    "top_features": [
      {"time_step": 28, "feature": "volume", "attribution": 0.35},
      {"time_step": 29, "feature": "close", "attribution": 0.28}
    ]
  },
  "completeness_error": 0.002,
  "model_accuracy": 0.58
}
```
- HTML报告：时间步×特征热力图（用HTML table+背景色模拟），中文，不出现代码

### 5. 输出文件（全部放到本文件夹 `task_b_lstm\`）
- `lstm_model.py` — LSTM模型定义+训练
- `ig_generic.py` — 通用IG实现（可被其他模型复用）
- `generate_ig_report.py` — 报告生成（运行入口）
- `lstm_checkpoint.pt` — 训练好的模型权重
- `ig_report_lstm.json` — JSON归因报告
- `ig_report_lstm.html` — HTML报告（中文，不出现代码）
- `README.md` — 完成后写README，包含：
  - 完成状态（DONE/INCOMPLETE）
  - 产出文件清单
  - 运行方式
  - 验证结果（完备性误差、归因矩阵形状）
  - akshare是否可用，如不可用说明用了什么替代

## 验证标准
- IG完备性误差<1%
- 归因矩阵形状 = (seq_len, n_features) = (30, 5)
- 关键时间步（接近预测点）归因值应较大
- 代码能独立运行
- ig_generic.py 能同时适用于LSTM和简单MLP

## 注意
- 只在本文件夹内写文件
- 代码必须能独立运行
- 如果akshare获取数据失败，用随机时序数据代替，但在README里说明
- 完成后在README.md里标DONE
