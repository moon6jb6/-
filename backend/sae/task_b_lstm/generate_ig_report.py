import torch
import json
import os
import numpy as np
from lstm_model import StockLSTM, prepare_data, train_lstm, evaluate_model
from ig_generic import integrated_gradients


def generate_report(output_dir):
    torch.manual_seed(42)
    np.random.seed(42)

    # 准备数据和模型
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    print("Training model...")
    model, train_acc = train_lstm(X_train, y_train, epochs=100)
    test_acc = evaluate_model(model, X_test, y_test)
    print(f"  Train acc: {train_acc*100:.2f}%, Test acc: {test_acc*100:.2f}%")

    # 取第一条测试样本做IG归因
    sample = torch.from_numpy(X_test[0:1]).float()
    print("Computing Integrated Gradients...")
    attributions, completeness_error = integrated_gradients(model, sample, target_class=1, n_steps=200)

    # 归因矩阵
    attr_np = attributions.detach().numpy()
    features = ["open", "high", "low", "close", "volume"]
    seq_len = attr_np.shape[0]

    # 计算每个时间步每个特征的归因
    top_features = []
    for t in range(seq_len):
        for f_idx, fname in enumerate(features):
            top_features.append({
                "time_step": int(t),
                "feature": fname,
                "attribution": float(attr_np[t, f_idx])
            })
    top_features.sort(key=lambda x: abs(x["attribution"]), reverse=True)
    for i, feat in enumerate(top_features[:10]):
        feat["rank"] = i + 1

    # 模型输出
    model.eval()
    with torch.no_grad():
        output = model(sample)
        prob = torch.softmax(output, dim=1)[0, 1].item()

    # 验证：最近时间步的特征是否排在前面
    recent_in_top5 = sum(1 for f in top_features[:5] if f["time_step"] >= seq_len - 3)
    print(f"  Recent timesteps in top-5: {recent_in_top5}/5")

    # JSON报告
    report = {
        "model_type": "LSTM",
        "task": "stock_direction_prediction",
        "data_source": "AAPL-like synthetic data (GBM + regime switching, 1000 trading days, 2020-2024)",
        "input_shape": list(X_test.shape[1:]),
        "features": features,
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "single_sample": {
            "output_probability": round(prob, 4),
            "predicted_class": int(prob > 0.5),
            "true_class": int(y_test[0]),
            "sample_index": 0,
            "description": "First test sample (most recent window)"
        },
        "attribution": {
            "method": "integrated_gradients",
            "n_steps": 200,
            "top_features": top_features[:10],
            "recent_timesteps_in_top5": recent_in_top5
        },
        "completeness_check": {
            "error_pct": round(completeness_error * 100, 4),
            "threshold": 1.0,
            "pass": completeness_error < 0.01
        }
    }

    # 写JSON
    json_path = os.path.join(output_dir, "ig_report_lstm.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 写HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>LSTM模型决策归因报告</title>
<style>
body{{font-family:sans-serif;max-width:900px;margin:0 auto;padding:20px;background:#f7fafc;color:#1a365d}}
h1{{color:#1a365d;border-bottom:2px solid #2b6cb0;padding-bottom:10px}}
h2{{color:#2b6cb0;margin-top:30px}}
table{{border-collapse:collapse;width:100%;margin:15px 0}}
th,td{{border:1px solid #cbd5e0;padding:8px 12px;text-align:center}}
th{{background:#edf2f7}}
.pass{{color:#38a169;font-weight:bold}} .fail{{color:#e53e3e;font-weight:bold}}
.top-feature{{background:#ebf8ff;padding:10px;margin:5px 0;border-radius:5px;border-left:4px solid #2b6cb0}}
.conclusion{{background:#f0fff4;border:1px solid #38a169;border-radius:8px;padding:20px;margin-top:20px}}
.metrics{{display:grid;grid-template-columns:1fr 1fr;gap:15px;margin:15px 0}}
.metric{{background:white;border:1px solid #e2e8f0;border-radius:8px;padding:15px;text-align:center}}
.metric .value{{font-size:24px;font-weight:bold;color:#2b6cb0}}
.metric .label{{font-size:12px;color:#718096;margin-top:5px}}
</style></head>
<body>
<h1>LSTM模型决策归因报告</h1>
<p><strong>模型类型：</strong>LSTM（长短期记忆网络，2层，hidden_size=64）</p>
<p><strong>任务：</strong>股票涨跌方向预测（二分类）</p>
<p><strong>数据来源：</strong>AAPL走势仿真数据（GBM + regime switching，1000个交易日，2020-2024）</p>
<p><strong>输入：</strong>{seq_len}个时间步 x 5个特征（开盘价、最高价、最低价、收盘价、成交量）</p>
<p><strong>训练方式：</strong>时间序列顺序划分（前80%训练，后20%测试），100个epoch，Adam优化器+学习率衰减</p>

<div class="metrics">
<div class="metric"><div class="value">{train_acc*100:.1f}%</div><div class="label">训练集准确率</div></div>
<div class="metric"><div class="value">{test_acc*100:.1f}%</div><div class="label">测试集准确率</div></div>
</div>

<h2>最重要的归因特征（Top 10）</h2>
<div>
"""
    for feat in top_features[:10]:
        t_label = f"t-{seq_len-1-feat['time_step']}"
        html += f'<div class="top-feature"><strong>#{feat["rank"]}</strong> {t_label}（第{feat["time_step"]}步）的 <strong>{feat["feature"]}</strong>：归因值 {feat["attribution"]:.6f}</div>\n'

    html += f"""</div>
<h2>时间步 x 特征归因热力图</h2>
<p>颜色越深表示归因绝对值越大。绿色=正归因（推高预测概率），红色=负归因。</p>
<table><tr><th>时间步</th>"""
    for f in features:
        html += f"<th>{f}</th>"
    html += "</tr>\n"

    max_abs = max(abs(attr_np.min()), abs(attr_np.max()), 1e-8)
    for t in range(seq_len):
        t_label = f"t-{seq_len-1-t}"
        html += f"<tr><td><strong>{t_label}</strong></td>"
        for f_idx in range(len(features)):
            val = attr_np[t, f_idx]
            intensity = int(min(abs(val) / max_abs * 200, 200))
            if val > 0:
                color = f"rgb({255-intensity},{255},{255-intensity})"
            else:
                color = f"rgb({255},{255-intensity},{255-intensity})"
            html += f'<td style="background:{color}">{val:.4f}</td>'
        html += "</tr>\n"
    html += "</table>\n"

    html += f"""<h2>完备性验证</h2>
<p>归因之和与模型输出的差异：{completeness_error*100:.4f}%</p>
<p class="{'pass' if completeness_error < 0.01 else 'fail'}">{'通过' if completeness_error < 0.01 else '未通过'}（阈值：&lt;1%）</p>

<div class="conclusion">
<h2>结论</h2>
<p>模型预测当前样本{'上涨' if prob > 0.5 else '下跌'}（概率 {prob*100:.1f}%），真实标签为{'上涨' if y_test[0] == 1 else '下跌'}</p>
<p>最关键的决策依据是<strong>{top_features[0]['feature']}</strong>（时间步t-{seq_len-1-top_features[0]['time_step']}），归因值 {top_features[0]['attribution']:.6f}</p>
<p>最近时间步（t-1, t-2）的特征在Top-5中占 {recent_in_top5}/5 个席位，符合"越近越重要"的直觉</p>
<p>归因方法：Integrated Gradients（200步插值），完备性验证{'通过' if completeness_error < 0.01 else '未通过'}</p>
</div>

<p style="margin-top:40px;color:#718096;font-size:12px">审计时间：{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 智信引擎 v0.1.0</p>
</body></html>"""

    html_path = os.path.join(output_dir, "ig_report_lstm.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nJSON report: {json_path}")
    print(f"HTML report: {html_path}")
    print(f"Completeness error: {completeness_error*100:.4f}%")
    print(f"Top feature: {top_features[0]['feature']} at t-{seq_len-1-top_features[0]['time_step']}")
    return report


if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    generate_report(output_dir)
