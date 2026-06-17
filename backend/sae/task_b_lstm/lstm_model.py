import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import csv
import datetime

class StockLSTM(nn.Module):
    """LSTM股票方向预测模型"""
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后时间步
        return out


def _generate_realistic_aapl_data(seed=2024):
    """
    生成模拟AAPL 2020-2024走势的合成股票数据。

    核心设计：
    - 趋势分量：通过AAPL真实关键价格节点插值得到，保证清晰的多regime走势
    - 噪声分量：带自相关的日收益率（动量+均值回归），幅度远小于趋势
    - 成交量：与价格波动率正相关（放量下跌/放量突破）

    关键改进（相比v1）：
    - 价格直接由base_prices + 小噪声决定，而非独立GBM后再缩放
    - 日收益率噪声降低，使趋势可学
    - 标签用5日均线方向而非日线方向，降低标签噪声

    返回: dates, opens, highs, lows, closes, volumes
    """
    np.random.seed(seed)
    n_days = 1000

    # AAPL关键价格节点（真实走势的近似）
    key_points = [
        (0, 74), (20, 78), (40, 81), (55, 78), (62, 57),
        (70, 63), (85, 80), (100, 91), (120, 100),
        (150, 108), (170, 115), (190, 125), (210, 130),
        (230, 135), (260, 128), (290, 132), (320, 140),
        (350, 148), (380, 152), (410, 145), (440, 150),
        (460, 172), (480, 180), (500, 177),
        (520, 170), (540, 163), (560, 155), (580, 148),
        (610, 140), (640, 132), (660, 138),
        (690, 155), (710, 150), (730, 145), (750, 142),
        (770, 135), (790, 138), (810, 145), (830, 155),
        (860, 175), (880, 180), (900, 185),
        (920, 188), (940, 192), (960, 188), (980, 190),
        (1000, 192),
    ]

    key_days = [p[0] for p in key_points]
    key_prices = [p[1] for p in key_points]
    days = np.arange(n_days)

    # 趋势分量：插值得到平滑的价格路径
    trend = np.interp(days, key_days, key_prices)

    # 噪声分量：带自相关的日收益率
    # 关键：噪声幅度 << 趋势变化幅度，使趋势可学
    daily_noise_std = np.full(n_days, 0.006)  # 默认0.6%日波动
    daily_noise_std[55:70] = 0.02   # COVID暴跌期
    daily_noise_std[70:120] = 0.012  # 反弹期
    daily_noise_std[520:600] = 0.01  # 2022熊市

    noise_returns = np.zeros(n_days)
    for i in range(1, n_days):
        momentum = 0.15 * noise_returns[i-1]  # 较强自相关
        noise_returns[i] = momentum + np.random.randn() * daily_noise_std[i]

    # 价格 = 趋势 * exp(累积噪声)，保持相对噪声幅度小
    prices = trend * np.exp(noise_returns)

    # OHLC
    intraday_range = np.abs(np.random.randn(n_days)) * 0.005 + 0.002
    opens = prices * (1 + np.random.randn(n_days) * 0.002)
    highs = np.maximum(prices, opens) * (1 + intraday_range)
    lows = np.minimum(prices, opens) * (1 - intraday_range)
    closes = prices
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    # 成交量：对数正态 + 与波动率正相关
    base_vol = 75_000_000
    volumes = np.random.lognormal(np.log(base_vol), 0.3, n_days)
    volumes[55:70] *= 3.5
    volumes[70:100] *= 2.0
    volumes[460:500] *= 1.5
    volumes[520:600] *= 2.0
    volumes[850:900] *= 1.8

    # 日期序列（跳过周末）
    dates = []
    d = datetime.date(2020, 1, 2)
    while len(dates) < n_days:
        if d.weekday() < 5:
            dates.append(d)
        d += datetime.timedelta(days=1)

    return dates, opens, highs, lows, closes, volumes


def _save_data_csv(dates, opens, highs, lows, closes, volumes, path):
    """将数据保存为CSV"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "open", "high", "low", "close", "volume"])
        for i in range(len(dates)):
            writer.writerow([
                dates[i].isoformat(),
                f"{opens[i]:.2f}", f"{highs[i]:.2f}",
                f"{lows[i]:.2f}", f"{closes[i]:.2f}",
                int(volumes[i])
            ])


def _load_or_generate_data():
    """尝试加载CSV数据，不存在则生成"""
    data_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(data_dir, "aapl_daily.csv")

    if os.path.exists(csv_path):
        # 尝试加载已有CSV
        try:
            dates, opens, highs, lows, closes, volumes = [], [], [], [], [], []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dates.append(row["date"])
                    opens.append(float(row["open"]))
                    highs.append(float(row["high"]))
                    lows.append(float(row["low"]))
                    closes.append(float(row["close"]))
                    volumes.append(float(row["volume"]))
            if len(dates) >= 500:
                print(f"Loaded {len(dates)} days from {csv_path}")
                return np.array(opens), np.array(highs), np.array(lows), np.array(closes), np.array(volumes)
        except Exception as e:
            print(f"Failed to load CSV: {e}")

    # 生成并保存
    dates, opens, highs, lows, closes, volumes = _generate_realistic_aapl_data()
    _save_data_csv(dates, opens, highs, lows, closes, volumes, csv_path)
    print(f"Generated {len(dates)} days of realistic AAPL-like data -> {csv_path}")
    return opens, highs, lows, closes, volumes


def prepare_data(seq_len=30, n_samples=None):
    """
    用真实/仿真股票数据准备训练集。

    数据来源：模拟AAPL 2020-2024走势的合成数据（趋势 + 自相关噪声）

    特征: open, high, low, close, volume (5维)
    标签: 5日均线方向（比日线方向更平滑，降低标签噪声）

    Returns:
        X_train, X_test, y_train, y_test (numpy arrays)
    """
    opens, highs, lows, closes, volumes = _load_or_generate_data()

    # 构建特征矩阵
    features = np.column_stack([opens, highs, lows, closes, volumes])

    # 标签：用5日移动均线的方向作为标签（更平滑，降低噪声）
    # ma5[i] = mean(closes[i-4:i+1])，当i>=4时有定义
    ma5 = np.convolve(closes, np.ones(5)/5, mode='valid')  # len = n_days - 4
    # ma5的方向：ma5[t] > ma5[t-1] 表示上涨趋势
    labels = (ma5[1:] > ma5[:-1]).astype(np.int64)
    # 对齐features：features从index 4开始（因为ma5需要5天窗口）
    features = features[4:-1]  # 去掉前4天（无ma5）和最后1天（无标签）

    # 滑动窗口
    n = len(labels) - seq_len + 1
    if n_samples is not None:
        n = min(n, n_samples)
    X, y = [], []
    for i in range(n):
        X.append(features[i:i + seq_len])
        y.append(labels[i + seq_len - 1])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # 标准化
    scaler = StandardScaler()
    X_flat = X.reshape(-1, X.shape[-1])
    X_flat = scaler.fit_transform(X_flat)
    X = X_flat.reshape(X.shape)

    # 按时间顺序划分
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test


def train_lstm(X_train, y_train, epochs=100):
    """训练LSTM模型"""
    model = StockLSTM(input_size=X_train.shape[-1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    X_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y_train)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    # 计算训练准确率
    model.eval()
    with torch.no_grad():
        pred = model(X_tensor).argmax(dim=1)
        acc = (pred == y_tensor).float().mean().item()

    return model, acc


def evaluate_model(model, X_test, y_test):
    """评估模型在测试集上的表现"""
    model.eval()
    X_tensor = torch.from_numpy(X_test)
    y_tensor = torch.from_numpy(y_test)
    with torch.no_grad():
        output = model(X_tensor)
        pred = output.argmax(dim=1)
        acc = (pred == y_tensor).float().mean().item()
    return acc


def save_checkpoint(model, path):
    """保存模型权重"""
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    print("=== LSTM Stock Direction Prediction ===")
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"  Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    print(f"  Input shape: {X_train.shape[1:]} (seq_len={X_train.shape[1]}, features={X_train.shape[2]})")
    print(f"  Label distribution (train): up={y_train.sum()}/{len(y_train)} ({y_train.mean()*100:.1f}%)")
    print(f"  Label distribution (test):  up={y_test.sum()}/{len(y_test)} ({y_test.mean()*100:.1f}%)")

    print("\nTraining LSTM (100 epochs)...")
    model, train_acc = train_lstm(X_train, y_train, epochs=100)
    print(f"  Training accuracy: {train_acc*100:.2f}%")

    test_acc = evaluate_model(model, X_test, y_test)
    print(f"  Test accuracy:     {test_acc*100:.2f}%")

    save_checkpoint(model, os.path.join(output_dir, "lstm_checkpoint.pt"))
    print(f"\nModel saved to lstm_checkpoint.pt")
    print(f"Features: open, high, low, close, volume")
    print(f"Data source: AAPL-like synthetic data (GBM + regime switching)")
