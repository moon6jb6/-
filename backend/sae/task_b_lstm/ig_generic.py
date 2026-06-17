import torch
import numpy as np

def integrated_gradients(model, input_tensor, baseline=None, n_steps=50, target_class=1):
    """
    通用Integrated Gradients，适用于任意PyTorch模型

    Args:
        model: PyTorch模型
        input_tensor: 输入tensor (batch, seq_len, features) 或 (batch, features)
        baseline: 基线输入，默认全零
        n_steps: 插值步数
        target_class: 目标类别

    Returns:
        attributions: 归因矩阵，与input_tensor同形状（去掉batch维度）
        completeness_error: 完备性误差（应<1%）
    """
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    # 保存原始batch维度
    original_shape = input_tensor.shape
    if input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(0)
        baseline = baseline.unsqueeze(0)
        added_batch = True
    else:
        added_batch = False

    scaled_inputs = [
        baseline + (float(i) / n_steps) * (input_tensor - baseline)
        for i in range(n_steps + 1)
    ]

    grads = []
    for x in scaled_inputs:
        x = x.clone().detach().requires_grad_(True)
        output = model(x)
        # 处理不同输出格式
        if output.dim() > 1 and output.shape[-1] > 1:
            target_output = output[:, target_class]
        else:
            target_output = output.squeeze(-1)
        target_output.sum().backward()
        grads.append(x.grad.detach())

    avg_grads = torch.stack(grads).mean(dim=0)
    attributions = (input_tensor - baseline) * avg_grads

    # 完备性验证
    with torch.no_grad():
        output_final = model(input_tensor)
        baseline_output = model(baseline)
        if output_final.dim() > 1 and output_final.shape[-1] > 1:
            expected = (output_final[:, target_class] - baseline_output[:, target_class]).item()
        else:
            expected = (output_final.squeeze(-1) - baseline_output.squeeze(-1)).item()
        actual = attributions.sum().item()
        if abs(expected) > 1e-8:
            completeness_error = abs(actual - expected) / abs(expected)
        else:
            completeness_error = abs(actual - expected)

    # 去掉batch维度
    attributions = attributions.squeeze(0)

    return attributions, completeness_error

def compute_attributions_batch(model, X, target_class=1, n_steps=50):
    """批量计算多个样本的归因"""
    all_attributions = []
    all_errors = []
    for i in range(len(X)):
        inp = torch.from_numpy(X[i:i+1]).float()
        attr, err = integrated_gradients(model, inp, target_class=target_class, n_steps=n_steps)
        all_attributions.append(attr.numpy())
        all_errors.append(err)
    return np.array(all_attributions), np.array(all_errors)
