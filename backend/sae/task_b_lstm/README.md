# 任务B完成报告

**状态：DONE**

## 产出文件
- lstm_model.py — LSTM模型定义+训练
- ig_generic.py — 通用IG实现
- generate_ig_report.py — 报告生成入口
- lstm_checkpoint.pt — 训练好的模型权重
- ig_report_lstm.json — JSON归因报告
- ig_report_lstm.html — HTML报告

## 运行方式
python generate_ig_report.py

## 验证结果
- 完备性误差：0.7071%
- 归因矩阵形状：(30, 5)
- 数据来源：随机时序数据（akshare不可用）
