import pandas as pd
import numpy as np
import os
import math
import torch
import torch.nn as nn
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from scipy.optimize import minimize
from contextlib import contextmanager
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import vectorbt as vbt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)

# -------------------------- 1. 数据预处理（保持不变） --------------------------
def prepare_data(data_path, feature_cols, target_col="y1", return_col="return"):
    data = pd.read_csv(data_path)
    print(f"原始数据形状：{data.shape}")
    
    data["original_index"] = data.index  # 保留原始索引用于对齐
    data = data.dropna(subset=[return_col]).copy()  # 移除return为空的行
    print(f"删除{return_col}列NaN后的数据形状：{data.shape}")
    
    # 处理特征极端异常值（3σ法则）
    for col in feature_cols:
        mean = data[col].mean()
        std = data[col].std()
        data[col] = np.clip(data[col], mean - 3*std, mean + 3*std)
    
    # 填充特征缺失值（中位数更抗异常值）
    feature_means = data[feature_cols].median()
    data[feature_cols] = data[feature_cols].fillna(feature_means)
    
    # 填充目标列缺失值（众数）
    if data[target_col].isna().any():
        target_mode = data[target_col].mode()[0]
        data[target_col] = data[target_col].fillna(target_mode)
        print(f"{target_col}列用众数({target_mode})填充缺失值")
    
    # 打印目标列类别分布（检查不平衡）
    class_dist = data[target_col].value_counts(normalize=True)
    print(f"目标列{target_col}类别分布：\n{class_dist}")
    if any(class_dist > 0.85):
        print("⚠️  警告：目标列类别严重不平衡，可能导致预测极端")
    
    data = data.reset_index(drop=True)
    print(f"最终数据形状：{data.shape}")
    return data, feature_cols

# -------------------------- 2. 滑动窗口生成（保持不变） --------------------------
def create_sliding_windows(
    stock_data,
    feature_cols,
    target_col="y1",
    return_col="return",
    seq_len=40,          # 时序序列长度（独立配置）
    scaler_len=50,       # 标准化拟合数据长度（独立配置）
    train_val_len=600,   # 训练+验证集总长度（独立配置）
    test_len=40,         # 测试集长度（独立配置）
    window_step=None,    # 窗口步长（默认=test_len，可独立设置）
    expected_gap=0,      # 窗口间间隙（步长=test_len+expected_gap，优先级低于window_step）
    split_type="val_at_end"  # 验证集分割方式
):
    total_days = len(stock_data)
    window_size = scaler_len + train_val_len + test_len  # 总窗口大小
    print(f"窗口配置：标准化{scaler_len}条 + 训练验证{train_val_len}条 + 测试{test_len}条 = 总{window_size}条 | 分割方式：{split_type}")

    # 参数校验
    if window_size <= 0:
        raise ValueError("窗口各部分长度需为正数")
    if seq_len <= 0:
        raise ValueError("seq_len（时序序列长度）需为正数")
    if total_days < window_size:
        raise ValueError(f"数据长度不足：总长度{total_days} < 窗口大小{window_size}")
    if test_len <= 0:
        raise ValueError("test_len（测试集长度）需为正数")
    if split_type not in ["val_at_end", "val_in_middle"]:
        raise ValueError("split_type仅支持 'val_at_end' 和 'val_in_middle'")

    # 确定窗口步长
    if window_step is None:
        window_step = test_len + expected_gap
    if window_step <= 0:
        raise ValueError("窗口步长需为正数")

    max_start_idx = total_days - window_size
    windows = []

    for start_idx in range(0, max_start_idx + 1, window_step):
        # 提取窗口数据
        window_data = stock_data.iloc[start_idx:start_idx + window_size].copy()
        scaler_data = window_data.iloc[:scaler_len].copy()
        train_val_data = window_data.iloc[scaler_len:scaler_len + train_val_len].copy()
        test_data = window_data.iloc[scaler_len + train_val_len:].copy()

        # 校验数据分割
        if len(scaler_data) != scaler_len or len(train_val_data) != train_val_len or len(test_data) != test_len:
            print(f"跳过窗口（起始索引{start_idx}）：数据分割异常")
            continue

        # 鲁棒标准化（减少异常值影响）
        feature_means = scaler_data[feature_cols].median()
        scaler = RobustScaler()
        scaler.fit(scaler_data[feature_cols].fillna(feature_means))

        # 时序数据生成函数
        def create_seq_data(data, task_target_col, history_data=None):
            X, y, original_indices = [], [], []
            combined_data = pd.concat([history_data, data], ignore_index=True) if history_data is not None else data
            data_len = len(data)

            for i in range(data_len):
                if history_data is not None:
                    end_idx = len(history_data) + i
                else:
                    end_idx = i + seq_len
                    if end_idx > len(combined_data):
                        break

                start_idx_seq = end_idx - seq_len
                start_idx_seq = max(start_idx_seq, 0)  # 避免越界

                seq_features = combined_data.iloc[start_idx_seq:end_idx][feature_cols].fillna(feature_means)
                seq_x = scaler.transform(seq_features)

                target_val = data.iloc[i][task_target_col]
                if not pd.isna(target_val):
                    X.append(seq_x)
                    y.append(target_val)
                    original_indices.append(data.iloc[i]["original_index"])

            return np.array(X), np.array(y), original_indices

        # 生成训练验证集数据（分类+回归）
        X_train_val_cls, y_train_val_cls, _ = create_seq_data(
            data=train_val_data, task_target_col=target_col, history_data=None
        )
        X_train_val_reg, y_train_val_reg, _ = create_seq_data(
            data=train_val_data, task_target_col=return_col, history_data=None
        )

        # 划分训练/验证集（两种分割方式）
        if split_type == "val_at_end":
            # 验证集在末尾（7:3分割）
            X_train_cls, X_val_cls, y_train_cls, y_val_cls = train_test_split(
                X_train_val_cls, y_train_val_cls, test_size=0.3, shuffle=False
            )
            X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(
                X_train_val_reg, y_train_val_reg, test_size=0.3, shuffle=False
            )
        else:
            # 验证集在中间（35%:30%:35% 分割）
            val_len = int(len(train_val_data) * 0.3)
            train1_len = int(len(train_val_data) * 0.35)
            train2_len = len(train_val_data) - train1_len - val_len

            # 分类任务分割
            X_train1_cls = X_train_val_cls[:train1_len]
            y_train1_cls = y_train_val_cls[:train1_len]
            X_val_cls = X_train_val_cls[train1_len:train1_len + val_len]
            y_val_cls = y_train_val_cls[train1_len:train1_len + val_len]
            X_train2_cls = X_train_val_cls[train1_len + val_len:]
            y_train2_cls = y_train_val_cls[train1_len + val_len:]
            X_train_cls = np.concatenate([X_train1_cls, X_train2_cls], axis=0)
            y_train_cls = np.concatenate([y_train1_cls, y_train2_cls], axis=0)

            # 回归任务分割
            X_train1_reg = X_train_val_reg[:train1_len]
            y_train1_reg = y_train_val_reg[:train1_len]
            X_val_reg = X_train_val_reg[train1_len:train1_len + val_len]
            y_val_reg = y_train_val_reg[train1_len:train1_len + val_len]
            X_train2_reg = X_train_val_reg[train1_len + val_len:]
            y_train2_reg = y_train_val_reg[train1_len + val_len:]
            X_train_reg = np.concatenate([X_train1_reg, X_train2_reg], axis=0)
            y_train_reg = np.concatenate([y_train1_reg, y_train2_reg], axis=0)

        # 生成测试集数据
        X_test_cls, y_test_cls, test_indices_cls = create_seq_data(
            data=test_data, task_target_col=target_col, history_data=train_val_data
        )
        X_test_reg, y_test_reg, test_indices_reg = create_seq_data(
            data=test_data, task_target_col=return_col, history_data=train_val_data
        )

        # 过滤无效窗口
        if len(X_train_cls) == 0 or len(X_test_cls) == 0:
            print(f"跳过窗口（起始索引{start_idx}）：分类数据为空")
            continue
        if len(X_train_reg) == 0 or len(X_test_reg) == 0:
            print(f"跳过窗口（起始索引{start_idx}）：回归数据为空")
            continue

        # 窗口数据结构
        windows.append({
            "cls": {
                "X_train": X_train_cls, "y_train": y_train_cls,
                "X_val": X_val_cls, "y_val": y_val_cls,
                "X_test": X_test_cls, "y_test": y_test_cls,
                "test_indices": test_indices_cls,
                "target_col": target_col
            },
            "reg": {
                "X_train": X_train_reg, "y_train": y_train_reg,
                "X_val": X_val_reg, "y_val": y_val_reg,
                "X_test": X_test_reg, "y_test": y_test_reg,
                "test_indices": test_indices_reg,
                "target_col": return_col
            },
            "scaler": scaler,
            "window_idx": len(windows),
            "data": stock_data.copy(),
            "split_type": split_type,
            "meta": {
                "start_idx": start_idx,
                "end_idx": start_idx + window_size - 1,
                "window_step": window_step
            }
        })

    print(f"生成窗口总数：{len(windows)}（步长：{window_step}）")
    return windows

# -------------------------- 3. BPNN模型（保持不变） --------------------------
class BPNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层→隐藏层
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层→输出层
        self.activation = nn.Tanh()  # 非线性激活函数

    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)  # 展平时序特征
        hidden = self.activation(self.bn1(self.fc1(x_flat)))
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        return output

# -------------------------- 4. 扩散模型（修正：增强修正权重） --------------------------
T = 300
def make_beta_schedule(T):
    s = 0.005
    t = torch.linspace(0, 1, T+1, device=device)
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    abars = alphas_cumprod[1:]
    abars_prev = alphas_cumprod[:-1]
    betas = 1 - abars / abars_prev
    return betas.clamp(1e-5, 0.015)  # 适当增大噪声上限

betas = make_beta_schedule(T)
alphas = 1.0 - betas
abars = torch.cumprod(alphas, dim=0)

class TimeEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
    
    def forward(self, t):
        return self.fc(t.view(-1, 1))

class EpsNet(nn.Module):
    def __init__(self, tdim=128, hidden=512):  # 增大隐藏层维度，增强学习能力
        super().__init__()
        self.temb = TimeEmbedding(tdim)
        self.mlp = nn.Sequential(
            nn.Linear(2 + tdim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden * 2), nn.SiLU(),
            nn.Linear(hidden * 2, hidden), nn.SiLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, r_t, cond, t):
        te = self.temb(t)
        x = torch.cat([r_t, cond, te], dim=-1)
        return self.mlp(x)

@torch.no_grad()
def ddim_sample(eps_model, predictor, xb, steps=30, alpha=0.2):  # 降低alpha，增强Diffusion修正权重
    xb = xb.to(device)
    B = xb.size(0)
    # 使用BPNN的logits（未sigmoid）作为输入，增大修正空间
    yhat_logits = predictor(xb).squeeze()  # 原始输出（logits）
    yhat = torch.sigmoid(yhat_logits)  # 仅用于概率参考

    idx = torch.linspace(T-1, 0, steps, dtype=torch.long, device=device)
    r_t = torch.zeros_like(yhat_logits)  # 基于logits的偏差初始化

    for i in range(steps):
        t = idx[i]
        a_bar_t = abars[t] if t < len(abars) else torch.tensor(1.0, device=device)
        a_bar_t_prev = abars[idx[i+1]] if i < steps-1 else torch.tensor(1.0, device=device)

        eps_hat = eps_model(r_t.unsqueeze(1), yhat.unsqueeze(1), torch.full((B,), float(t), device=device))
        sqrt_a_bar_t = torch.sqrt(a_bar_t)
        sqrt_one_minus_a_bar_t = torch.sqrt(1.0 - a_bar_t)
        x0 = (r_t.unsqueeze(1) - sqrt_one_minus_a_bar_t * eps_hat) / (sqrt_a_bar_t + 1e-8)

        sqrt_a_bar_t_prev = torch.sqrt(a_bar_t_prev)
        r_t = (sqrt_a_bar_t_prev * x0 + torch.sqrt(1.0 - a_bar_t_prev) * eps_hat).squeeze()

    # 基于logits修正，再转换为概率
    corrected_logits = alpha * yhat_logits + (1 - alpha) * x0.squeeze()
    final_prob = torch.sigmoid(corrected_logits)
    return final_prob

# -------------------------- 5. 训练工具（EMA，保持不变） --------------------------
class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {n: p.data.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}
    
    def update(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data
    
    def apply_shadow(self, model):
        self.backup = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data = self.shadow[n].clone()
    
    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[n].clone()

@contextmanager
def use_ema(model, ema_obj):
    ema_obj.apply_shadow(model)
    try:
        yield
    finally:
        ema_obj.restore(model)

# -------------------------- 6. BPNN训练函数（保持不变） --------------------------
def train_bpnn_predictor(model, tr_loader, va_loader, epochs=60):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    # 计算类别权重（解决不平衡）
    train_labels = []
    for _, yb in tr_loader:
        train_labels.extend(yb.numpy())
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=class_weights[1]/class_weights[0])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    ema_pred = EMA(model, decay=0.995)

    best_val_loss = float("inf")
    best_state, best_shadow = None, None

    for ep in range(epochs):
        model.train()
        train_loss = []
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            pred = model(xb).squeeze()
            loss = crit(pred, yb)
            
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            ema_pred.update(model)
            train_loss.append(loss.item())
        
        model.eval()
        val_loss = []
        with use_ema(model, ema_pred), torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                pred = model(xb).squeeze()
                val_loss.append(crit(pred, yb).item())
        
        avg_train = np.mean(train_loss)
        avg_val = np.mean(val_loss)
        print(f"[BPNN] epoch {ep+1:03d} | train {avg_train:.6f} | val {avg_val:.6f}")
        scheduler.step()
        
        if avg_val < best_val_loss - 1e-6:
            best_val_loss = avg_val
            best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
            best_shadow = {k: v.clone() for k, v in ema_pred.shadow.items()}
    
    model.load_state_dict(best_state)
    ema_pred.shadow = best_shadow
    return model, ema_pred, best_state

# -------------------------- 新增：7. 纯BPNN模型训练函数（无LightGBM融合） --------------------------
def train_bpnn_single(windows, feature_dim, hidden_size=32, save_root="results", epochs=60, batch_size=32, model_suffix=""):
    model_name = f"bpnn_single{model_suffix}"
    all_pred_cols = [f"{model_name}_prob_pred"]  # 纯BPNN预测列名
    
    for window in windows:
        window["data"][all_pred_cols[0]] = np.nan
    
    for window_idx, window in enumerate(windows):
        print(f"\n=== 纯BPNN{model_suffix} - 窗口 {window_idx + 1}/{len(windows)} ===")
        split_type = window["split_type"]
        cls_data = window["cls"]
        df = window["data"]
        target_col = cls_data["target_col"]
        
        model_dir, pred_dir = create_save_dirs(save_root, model_name, split_type, window_idx)
        
        # 准备BPNN训练/验证数据
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(cls_data["X_train"], dtype=torch.float32),
            torch.tensor(cls_data["y_train"], dtype=torch.float32)
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(cls_data["X_val"], dtype=torch.float32),
            torch.tensor(cls_data["y_val"], dtype=torch.float32)
        )
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # BPNN输入维度 = 序列长度 × 特征维度
        bpnn_input_size = cls_data["X_train"].shape[1] * cls_data["X_train"].shape[2]
        bpnn_model = BPNNModel(input_size=bpnn_input_size, hidden_size=hidden_size)
        bpnn_model, ema_bpnn, best_bpnn_state = train_bpnn_predictor(bpnn_model, train_loader, val_loader, epochs=epochs)
        
        # 纯BPNN预测（带EMA）
        with use_ema(bpnn_model, ema_bpnn), torch.no_grad():
            y_bpnn_test = torch.sigmoid(bpnn_model(torch.tensor(cls_data["X_test"], dtype=torch.float32).to(device))).cpu().numpy().squeeze()
        
        # 评估预测效果
        acc = accuracy_score(cls_data["y_test"], (y_bpnn_test>0.5).astype(int))
        auc = roc_auc_score(cls_data["y_test"], y_bpnn_test)
        brier = brier_score_loss(cls_data["y_test"], y_bpnn_test)
        print(f"测试集准确率：{acc:.4f} | AUC：{auc:.4f} | Brier分数：{brier:.4f}")
        
        # 保存结果
        save_window_predictions(
            pred_dir=pred_dir,
            indices=cls_data["test_indices"],
            true_values=cls_data["y_test"],
            pred_values=y_bpnn_test,
            task_type="cls",
            target_col=target_col
        )
        save_model_files(model_dir, best_bpnn_state, model_type="torch", suffix="_bpnn_single")
        
        # 记录预测结果到窗口数据
        mask = df["original_index"].isin(cls_data["test_indices"])
        df.loc[mask, all_pred_cols[0]] = y_bpnn_test
        window["data"] = df
    
    final_df = pd.concat([window["data"] for window in windows], ignore_index=True).drop_duplicates("original_index")
    return final_df, all_pred_cols

# -------------------------- 8. 权重优化（修正：为BPNN设置权重下限） --------------------------
def optimize_hybrid_weights(y_val, y_lgb_val, y_bpnn_val):
    """按文档逻辑优化权重：min ∑(y_t - w1*y_lgb - w2*y_bpnn)^2，约束w1+w2=1且w2≥0.2"""
    def loss_fn(weights):
        w1, w2 = weights
        y_hybrid = w1 * y_lgb_val + w2 * y_bpnn_val
        return mean_squared_error(y_val, y_hybrid)
    
    # 初始权重（等权）
    initial_weights = [0.5, 0.5]
    # 约束条件：w1 + w2 = 1
    constraints = [{'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1}]
    # 权重范围：w2≥0.2（确保BPNN/Diffusion有最低贡献）
    bounds = [(0.0, 0.8), (0.2, 1.0)]
    
    # 最小化损失函数
    result = minimize(loss_fn, initial_weights, constraints=constraints, bounds=bounds, method='SLSQP')
    return result.x[0], result.x[1]

# -------------------------- 9. 保存工具函数（保持不变） --------------------------
def create_save_dirs(save_root, model_name, split_type, window_idx):
    base_dir = os.path.join(save_root, split_type, model_name)
    model_dir = os.path.join(base_dir, "models", f"window_{window_idx}")
    pred_dir = os.path.join(base_dir, "predictions", f"window_{window_idx}")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    return model_dir, pred_dir

def save_window_predictions(pred_dir, indices, true_values, pred_values, task_type, target_col):
    df = pd.DataFrame({
        "original_index": indices,
        f"true_{target_col}": true_values,
        f"pred_{target_col}": pred_values
    })
    df.to_csv(os.path.join(pred_dir, f"window_predictions.csv"), index=False)
    print(f"窗口预测结果保存至：{pred_dir}")

def save_model_files(model_dir, model, model_type, suffix=""):
    if model_type == "torch":
        torch.save(model, os.path.join(model_dir, f"{model_type}_model{suffix}.pth"))
    elif model_type == "lgb":
        model.save_model(os.path.join(model_dir, f"{model_type}_model{suffix}.txt"))
    print(f"模型文件保存至：{model_dir}")

# -------------------------- 10. 模型训练与保存（保持不变，新增纯BPNN调用） --------------------------
# 10.1 线性回归（保持不变）
def train_linear_regression(windows, save_root="results", model_suffix=""):
    model_name = f"linear_regression{model_suffix}"
    all_pred_cols = [f"{model_name}_return_pred"]
    
    for window in windows:
        window["data"][all_pred_cols[0]] = np.nan
    
    for window_idx, window in enumerate(windows):
        print(f"\n=== 线性回归{model_suffix} - 窗口 {window_idx + 1}/{len(windows)} ===")
        split_type = window["split_type"]
        reg_data = window["reg"]
        df = window["data"]
        target_col = reg_data["target_col"]
        
        model_dir, pred_dir = create_save_dirs(save_root, model_name, split_type, window_idx)
        
        lr = LinearRegression()
        X_train_flat = reg_data["X_train"].reshape(reg_data["X_train"].shape[0], -1)
        lr.fit(X_train_flat, reg_data["y_train"])
        
        X_test_flat = reg_data["X_test"].reshape(reg_data["X_test"].shape[0], -1)
        test_pred = lr.predict(X_test_flat)
        test_mse = mean_squared_error(reg_data["y_test"], test_pred)
        print(f"测试集MSE: {test_mse:.4f}")
        
        save_window_predictions(
            pred_dir=pred_dir,
            indices=reg_data["test_indices"],
            true_values=reg_data["y_test"],
            pred_values=test_pred,
            task_type="reg",
            target_col=target_col
        )
        
        coef_df = pd.DataFrame({
            "feature": [f"feat_{i}" for i in range(X_train_flat.shape[1])],
            "coefficient": lr.coef_
        })
        coef_df.to_csv(os.path.join(model_dir, "linear_coefficients.csv"), index=False)
        
        mask = df["original_index"].isin(reg_data["test_indices"])
        df.loc[mask, all_pred_cols[0]] = test_pred
        window["data"] = df
    
    final_df = pd.concat([window["data"] for window in windows], ignore_index=True).drop_duplicates("original_index")
    return final_df, all_pred_cols

# 10.2 LightGBM-BPNN融合模型（保持不变）
def train_lgb_bpnn_fusion(windows, feature_dim, hidden_size=32, save_root="results", epochs=60, batch_size=32, model_suffix=""):
    model_name = f"lgb_bpnn_fusion{model_suffix}"
    all_pred_cols = [f"{model_name}_prob_pred"]
    
    for window in windows:
        window["data"][all_pred_cols[0]] = np.nan
    
    for window_idx, window in enumerate(windows):
        print(f"\n=== LightGBM-BPNN{model_suffix} - 窗口 {window_idx + 1}/{len(windows)} ===")
        split_type = window["split_type"]
        cls_data = window["cls"]
        df = window["data"]
        target_col = cls_data["target_col"]
        
        model_dir, pred_dir = create_save_dirs(save_root, model_name, split_type, window_idx)
        
        # 训练LightGBM
        X_train_flat = cls_data["X_train"].reshape(cls_data["X_train"].shape[0], -1)
        X_val_flat = cls_data["X_val"].reshape(cls_data["X_val"].shape[0], -1)
        X_test_flat = cls_data["X_test"].reshape(cls_data["X_test"].shape[0], -1)
        
        lgb_params = {
            "objective": "binary", "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.02,
            "num_leaves": 15,
            "max_depth": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "feature_fraction": 0.8,
            "verbose": -1
        }
        train_lgb = lgb.Dataset(X_train_flat, label=cls_data["y_train"])
        val_lgb = lgb.Dataset(X_val_flat, label=cls_data["y_val"], reference=train_lgb)
        lgb_model = lgb.train(lgb_params, train_lgb, num_boost_round=50, valid_sets=[val_lgb])
        
        y_lgb_val = lgb_model.predict(X_val_flat)
        y_lgb_test = lgb_model.predict(X_test_flat)
        
        # 训练BPNN
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(cls_data["X_train"], dtype=torch.float32),
            torch.tensor(cls_data["y_train"], dtype=torch.float32)
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(cls_data["X_val"], dtype=torch.float32),
            torch.tensor(cls_data["y_val"], dtype=torch.float32)
        )
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        bpnn_input_size = cls_data["X_train"].shape[1] * cls_data["X_train"].shape[2]
        bpnn_model = BPNNModel(input_size=bpnn_input_size, hidden_size=hidden_size)
        bpnn_model, ema_bpnn, best_bpnn_state = train_bpnn_predictor(bpnn_model, train_loader, val_loader, epochs=epochs)
        
        # BPNN预测
        with use_ema(bpnn_model, ema_bpnn), torch.no_grad():
            y_bpnn_val = torch.sigmoid(bpnn_model(torch.tensor(cls_data["X_val"], dtype=torch.float32).to(device))).cpu().numpy().squeeze()
            y_bpnn_test = torch.sigmoid(bpnn_model(torch.tensor(cls_data["X_test"], dtype=torch.float32).to(device))).cpu().numpy().squeeze()
        
        # 优化融合权重
        w1, w2 = optimize_hybrid_weights(cls_data["y_val"], y_lgb_val, y_bpnn_val)
        print(f"混合权重：LightGBM={w1:.3f}, BPNN={w2:.3f}")
        
        # 混合模型预测
        y_hybrid_test = w1 * y_lgb_test + w2 * y_bpnn_test
        acc = accuracy_score(cls_data["y_test"], (y_hybrid_test>0.5).astype(int))
        auc = roc_auc_score(cls_data["y_test"], y_hybrid_test)
        brier = brier_score_loss(cls_data["y_test"], y_hybrid_test)
        print(f"测试集准确率：{acc:.4f} | AUC：{auc:.4f} | Brier分数：{brier:.4f}")
        
        # 保存结果
        save_window_predictions(
            pred_dir=pred_dir,
            indices=cls_data["test_indices"],
            true_values=cls_data["y_test"],
            pred_values=y_hybrid_test,
            task_type="cls",
            target_col=target_col
        )
        save_model_files(model_dir, lgb_model, model_type="lgb")
        save_model_files(model_dir, best_bpnn_state, model_type="torch", suffix="_bpnn")
        
        # 记录预测结果
        mask = df["original_index"].isin(cls_data["test_indices"])
        df.loc[mask, all_pred_cols[0]] = y_hybrid_test
        window["data"] = df
    
    final_df = pd.concat([window["data"] for window in windows], ignore_index=True).drop_duplicates("original_index")
    return final_df, all_pred_cols

# 10.3 LightGBM-BPNN-Diffusion融合模型（核心修正）
def train_lgb_bpnn_diffusion_fusion(windows, feature_dim, hidden_size=32, save_root="results", epochs=60, batch_size=32, model_suffix=""):
    model_name = f"lgb_bpnn_diffusion_fusion{model_suffix}"
    all_pred_cols = [f"{model_name}_prob_pred"]
    
    for window in windows:
        window["data"][all_pred_cols[0]] = np.nan
    
    for window_idx, window in enumerate(windows):
        print(f"\n=== LightGBM-BPNN-Diffusion{model_suffix} - 窗口 {window_idx + 1}/{len(windows)} ===")
        split_type = window["split_type"]
        cls_data = window["cls"]
        df = window["data"]
        target_col = cls_data["target_col"]
        
        model_dir, pred_dir = create_save_dirs(save_root, model_name, split_type, window_idx)
        
        # 步骤1：训练LightGBM
        X_train_flat = cls_data["X_train"].reshape(cls_data["X_train"].shape[0], -1)
        X_val_flat = cls_data["X_val"].reshape(cls_data["X_val"].shape[0], -1)
        X_test_flat = cls_data["X_test"].reshape(cls_data["X_test"].shape[0], -1)
        
        lgb_params = {
            "objective": "binary", "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.02,
            "num_leaves": 15,
            "max_depth": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "feature_fraction": 0.8,
            "verbose": -1
        }
        train_lgb = lgb.Dataset(X_train_flat, label=cls_data["y_train"])
        val_lgb = lgb.Dataset(X_val_flat, label=cls_data["y_val"], reference=train_lgb)
        lgb_model = lgb.train(lgb_params, train_lgb, num_boost_round=50, valid_sets=[val_lgb])
        y_lgb_val = lgb_model.predict(X_val_flat)
        y_lgb_test = lgb_model.predict(X_test_flat)
        
        # 步骤2：训练BPNN
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(cls_data["X_train"], dtype=torch.float32),
            torch.tensor(cls_data["y_train"], dtype=torch.float32)
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(cls_data["X_val"], dtype=torch.float32),
            torch.tensor(cls_data["y_val"], dtype=torch.float32)
        )
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        bpnn_input_size = cls_data["X_train"].shape[1] * cls_data["X_train"].shape[2]
        bpnn_model = BPNNModel(input_size=bpnn_input_size, hidden_size=hidden_size)
        bpnn_model, ema_bpnn, best_bpnn_state = train_bpnn_predictor(bpnn_model, train_loader, val_loader, epochs=epochs//2)
        
        # 步骤3：训练Diffusion修正BPNN预测（核心修正）
        diffusion_train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(cls_data["X_train"], dtype=torch.float32),
                torch.tensor(cls_data["y_train"], dtype=torch.float32)
            ),
            batch_size=batch_size, shuffle=True
        )
        eps_model = EpsNet(tdim=128, hidden=512)  # 增大隐藏层
        opt = torch.optim.AdamW(eps_model.parameters(), lr=5e-4, weight_decay=1e-5)  # 降低学习率
        ema_diff = EMA(eps_model, decay=0.995)
        mse = nn.MSELoss()
        
        best_diff_loss = float("inf")
        best_eps_state = None
        
        for ep in range(epochs):
            eps_model.train()
            train_loss = []
            for xb, yb in diffusion_train_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                with torch.no_grad():
                    # 使用BPNN的logits计算偏差（方差更大）
                    yhat_logits = bpnn_model(xb).squeeze()
                    yhat = torch.sigmoid(yhat_logits)
                r0 = (yb - yhat_logits).unsqueeze(1)  # 基于logits的偏差（范围更大）
                
                t = torch.randint(0, T, (xb.size(0),), device=device).float()
                abar_t = abars[t.long()].unsqueeze(1)
                eps = torch.randn_like(r0)
                r_t = torch.sqrt(abar_t) * r0 + torch.sqrt(1 - abar_t) * eps
                
                eps_hat = eps_model(r_t, yhat.unsqueeze(1), t)
                loss = mse(eps_hat, eps)
                train_loss.append(loss.item())
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                ema_diff.update(eps_model)
            
            avg_loss = np.mean(train_loss)
            print(f"[Diffusion] epoch {ep+1}/{epochs} | loss: {avg_loss:.6f} | 最佳损失: {best_diff_loss:.6f}")
            if avg_loss < best_diff_loss - 1e-6:  # 严格损失下降才更新
                best_diff_loss = avg_loss
                best_eps_state = {k: v.clone() for k, v in eps_model.state_dict().items()}
        
        # 步骤4：Diffusion修正BPNN预测（验证修正效果）
        eps_model.load_state_dict(best_eps_state)
        with use_ema(bpnn_model, ema_bpnn), torch.no_grad():
            # 验证集修正（用于权重优化）
            y_bpnn_val_raw = torch.sigmoid(bpnn_model(torch.tensor(cls_data["X_val"], dtype=torch.float32).to(device))).cpu().numpy().squeeze()
            y_bpnn_val_diff = ddim_sample(eps_model, bpnn_model, torch.tensor(cls_data["X_val"], dtype=torch.float32)).cpu().numpy()
            
            # 测试集修正
            y_bpnn_test_raw = torch.sigmoid(bpnn_model(torch.tensor(cls_data["X_test"], dtype=torch.float32).to(device))).cpu().numpy().squeeze()
            y_bpnn_test_diff = ddim_sample(eps_model, bpnn_model, torch.tensor(cls_data["X_test"], dtype=torch.float32)).cpu().numpy()
        
        # 验证修正效果：打印修正前后的平均差异
        val_diff_mean = np.mean(np.abs(y_bpnn_val_diff - y_bpnn_val_raw))
        test_diff_mean = np.mean(np.abs(y_bpnn_test_diff - y_bpnn_test_raw))
        print(f"[Diffusion效果验证] 验证集修正差异：{val_diff_mean:.4f} | 测试集修正差异：{test_diff_mean:.4f}")
        if val_diff_mean < 0.01:
            print("⚠️  Diffusion在验证集修正效果微弱，建议增大epochs或调整模型结构")
        
        # 步骤5：优化混合权重（修正后BPNN+LightGBM）
        w1, w2 = optimize_hybrid_weights(cls_data["y_val"], y_lgb_val, y_bpnn_val_diff)
        print(f"混合权重：LightGBM={w1:.3f}, BPNN(Diffusion)={w2:.3f}")
        
        # 步骤6：混合模型预测
        y_hybrid_test = w1 * y_lgb_test + w2 * y_bpnn_test_diff
        acc = accuracy_score(cls_data["y_test"], (y_hybrid_test>0.5).astype(int))
        auc = roc_auc_score(cls_data["y_test"], y_hybrid_test)
        brier = brier_score_loss(cls_data["y_test"], y_hybrid_test)
        print(f"测试集准确率：{acc:.4f} | AUC：{auc:.4f} | Brier分数：{brier:.4f}")
        
        # 保存结果
        save_window_predictions(
            pred_dir=pred_dir,
            indices=cls_data["test_indices"],
            true_values=cls_data["y_test"],
            pred_values=y_hybrid_test,
            task_type="cls",
            target_col=target_col
        )
        save_model_files(model_dir, lgb_model, model_type="lgb")
        save_model_files(model_dir, best_bpnn_state, model_type="torch", suffix="_bpnn")
        save_model_files(model_dir, best_eps_state, model_type="torch", suffix="_diffusion")
        
        # 记录预测结果
        mask = df["original_index"].isin(cls_data["test_indices"])
        df.loc[mask, all_pred_cols[0]] = y_hybrid_test
        window["data"] = df
    
    final_df = pd.concat([window["data"] for window in windows], ignore_index=True).drop_duplicates("original_index")
    return final_df, all_pred_cols


def plot_if_nav_comparison(data, name_list=None, colors=None, title=None):
    """
    绘制IF沪深300股指期货模型净值对比图（支持8列模型+时分秒时间标签）
    
    参数：
    data: DataFrame，index为DatetimeIndex（支持时分秒），含'Close_price'和模型列
    name_list: 8个模型列名称列表（默认含2个BPnn单独模型）
    colors: 8个线条颜色（默认高区分度配色）
    title: 图表标题（可选）
    """
    # -------------------------- 1. 默认参数（8列模型）
    if name_list is None:
        name_list = [
            # 原有6列
            'linear_regression_val_end_return_pred',
            'lgb_bpnn_fusion_val_end_prob_pred',
            'lgb_bpnn_diffusion_fusion_val_end_prob_pred',
            'linear_regression_val_middle_return_pred',
            'lgb_bpnn_fusion_val_middle_prob_pred',
            'lgb_bpnn_diffusion_fusion_val_middle_prob_pred',
            # 新增2个BPnn单独模型（按实际列名修改！）
            'bpnn_val_end_prob_pred',
            'bpnn_val_middle_prob_pred'
        ]
    
    if colors is None:
        colors = ['#00A1FF', '#5ed935', '#f8ba00', '#ff2501', '#d31876', '#919292', '#0072BC', '#70AD47']
    
    if title is None:
        title = 'NAV Comparison of Eight Models for IF(China Securities Index 300 Futures)'
    
    # 校验：列数一致+列存在+时间index
    if len(name_list) != len(colors):
        raise ValueError(f"模型列数（{len(name_list)}）需与颜色数（{len(colors)}）一致")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data.index必须是DatetimeIndex，请先执行data.index = pd.to_datetime(data.index)")
    for col in name_list + ['Close_price']:
        if col not in data.columns:
            raise ValueError(f"data缺少列：{col}")
    
    # -------------------------- 2. 绘图样式
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": ["Arial", "Times New Roman"],
        "axes.unicode_minus": False,
        "lines.linewidth": 1.0,
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "xtick.labelsize": 9,  # 适配时分秒标签，适当缩小
        "ytick.labelsize": 10,
        "axes.labelsize": 14,
        "axes.titlesize": 16
    })
    
    # -------------------------- 3. 计算8个模型的净值
    close = data['Close_price']
    all_navs = []
    start_timestamps = []
    
    for col_name in name_list:
        pf = vbt.Portfolio.from_signals(
            close,
            entries=data[col_name] >= data[col_name].quantile(0.2, interpolation='higher'),
            exits=data[col_name] >= data[col_name].quantile(0.3, interpolation='higher'),
            short_entries=data[col_name] <= data[col_name].quantile(0.2, interpolation='lower'),
            short_exits=data[col_name] <= data[col_name].quantile(0.3, interpolation='lower'),
            upon_dir_conflict='adjacent',
            upon_opposite_entry='close',
            sl_stop=0.01,
            tp_stop=0.01,
        )
        
        # 兼容VectorBT版本
        try:
            nav_raw = pf.value()
        except:
            try:
                nav_raw = pf.total_value
            except:
                nav_raw = pf.value
        
        # 绑定原始时间index
        if len(nav_raw.shape) > 1:
            nav_raw = nav_raw.flatten()
        nav = pd.Series(nav_raw, index=data.index)
        
        # 记录净值起始时间
        non_constant_mask = nav != 100
        first_change_idx = nav[non_constant_mask].index[0] if non_constant_mask.any() else nav.index[0]
        start_timestamps.append(first_change_idx)
        all_navs.append(nav)
    
    unified_start = max(start_timestamps) if start_timestamps else data.index[0]
    print(f"统一切入时间：{unified_start}（切除前期100不变段）")
    
    # -------------------------- 4. 绘图（适配时分秒x标签）
    plt.figure(figsize=(14, 7))  # 加宽画布，避免时间标签拥挤
    
    for idx, (col_name, nav) in enumerate(zip(name_list, all_navs)):
        nav_clean = nav.loc[unified_start:]
        # 简化图例
        legend_name = (col_name.replace('_', ' ')
                      .replace('prob pred', '')
                      .replace('return pred', '')
                      .replace('val ', '')
                      .replace('middle', 'Mid')
                      .replace('end', 'End')
                      .strip())
        
        plt.plot(nav_clean.index, nav_clean.values, color=colors[idx], label=legend_name, alpha=0.9)
    
    # -------------------------- 5. 关键：适配时分秒的x轴标签
    plt.title(title, fontweight='bold')
    plt.xlabel('Datetime', fontsize=12)  # 标签改为Datetime（含时分秒）
    plt.ylabel('Net Asset Value', fontsize=12)
    
    ax = plt.gca()
    # 最多显示8个标签，避免拥挤
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    # 日期格式改为"年-月-日 时:分"（适配你的数据格式）
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=60, ha='right')  # 旋转60度，避免时分秒标签重叠
    
    # 图例适配8个条目
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        fontsize=9,
        frameon=True,
        facecolor='white',
        edgecolor='gray'
    )
    
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()