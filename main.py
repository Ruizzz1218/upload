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
import fusion_main as mm
from factor_pac import factors as factor


data = pd.read_csv('df_futures.csv')
calculator = factor.FactorCalculator(data)
data_features = calculator.get_combined_df()
data_features.to_csv('df_futures_with_features.csv')
# Configuration parameters (adjust according to data)
data_path = "df_futures_with_features.csv"  # Data path
target_col = "y1"                               # Classification target column
return_col = "return"                           # Regression target column

# Feature columns (replace with actual features)
feature_cols = ['group_volume', 'group_turnover', 'group_time', 'return', 'volume_spike', 
            'price_jump_probability', 'trade_continuation','relative_volatility_index', 
            'MFI', 'RSI', 'SRSI', 'PVI', 'VHF', 'CVI','CMO', 'VIDYA', 'UI', 'ACD', 
            'TS', 'MACD', 'VMACD', 'TMACD', 'QST','KVO', 'KVO_signal']

# Window parameters (fully configurable independently)
seq_len = 40          # Time series sequence length
scaler_len = 50       # Normalization data length
train_val_len = 600   # Total length of train + validation set
test_len = 40         # Test set length
window_step = 40      # Window step size

# Training parameters
epochs = 80  # Increase Diffusion training epochs
batch_size = 32
save_root = "8_model_results_bpnn_hybrid_revised_000014"  # Result saving directory (updated for 8 models)

# 1. Data preprocessing
data, feature_cols = mm.prepare_data(
    data_path=data_path,
    feature_cols=feature_cols,
    target_col=target_col,
    return_col=return_col
)
feature_dim = len(feature_cols)
print(f"Feature dimension: {feature_dim}")

# 2. Generate windows for two validation modes
print("\n===== Generate windows with [validation set at the end] =====")
windows_val_end = mm.create_sliding_windows(
    stock_data=data,
    feature_cols=feature_cols,
    target_col=target_col,
    return_col=return_col,
    seq_len=seq_len,
    scaler_len=scaler_len,
    train_val_len=train_val_len,
    test_len=test_len,
    window_step=window_step,
    split_type="val_at_end"
)

print("\n===== Generate windows with [validation set in the middle] =====")
windows_val_middle = mm.create_sliding_windows(
    stock_data=data,
    feature_cols=feature_cols,
    target_col=target_col,
    return_col=return_col,
    seq_len=seq_len,
    scaler_len=scaler_len,
    train_val_len=train_val_len,
    test_len=test_len,
    window_step=window_step,
    split_type="val_in_middle"
)

# Check window validity
if not windows_val_end and not windows_val_middle:
    print("No valid windows generated, program exits")
    exit()

# 3. Train 4 models with [validation set at the end] (added single BPNN)
print("\n===== Train 4 models with [validation set at the end] =====")
lr_val_end_df, lr_val_end_cols = mm.train_linear_regression(windows_val_end, save_root=save_root, model_suffix="_val_end")
bpnn_single_val_end_df, bpnn_single_val_end_cols = mm.train_bpnn_single(  # Added single BPNN training
    windows_val_end, feature_dim, hidden_size=32, save_root=save_root, epochs=epochs, batch_size=batch_size, model_suffix="_val_end"
)
lgb_bpnn_val_end_df, lgb_bpnn_val_end_cols = mm.train_lgb_bpnn_fusion(
    windows_val_end, feature_dim, hidden_size=32, save_root=save_root, epochs=epochs, batch_size=batch_size, model_suffix="_val_end"
)
lgb_bpnn_diff_val_end_df, lgb_bpnn_diff_val_end_cols = mm.train_lgb_bpnn_diffusion_fusion(
    windows_val_end, feature_dim, hidden_size=32, save_root=save_root, epochs=epochs, batch_size=batch_size, model_suffix="_val_end"
)

# 4. Train 4 models with [validation set in the middle] (added single BPNN)
print("\n===== Train 4 models with [validation set in the middle] =====")
lr_val_mid_df, lr_val_mid_cols = mm.train_linear_regression(windows_val_middle, save_root=save_root, model_suffix="_val_middle")
bpnn_single_val_mid_df, bpnn_single_val_mid_cols = mm.train_bpnn_single(  # Added single BPNN training
    windows_val_middle, feature_dim, hidden_size=32, save_root=save_root, epochs=epochs, batch_size=batch_size, model_suffix="_val_middle"
)
lgb_bpnn_val_mid_df, lgb_bpnn_val_mid_cols = mm.train_lgb_bpnn_fusion(
    windows_val_middle, feature_dim, hidden_size=32, save_root=save_root, epochs=epochs, batch_size=batch_size, model_suffix="_val_middle"
)
lgb_bpnn_diff_val_mid_df, lgb_bpnn_diff_val_mid_cols = mm.train_lgb_bpnn_diffusion_fusion(
    windows_val_middle, feature_dim, hidden_size=32, save_root=save_root, epochs=epochs, batch_size=batch_size, model_suffix="_val_middle"
)

# Key modification: Update all_dfs to include outputs of 8 models (2 validation modes × 4 models)
all_dfs = [
    # 4 models with validation set at the end
    (lr_val_end_df, lr_val_end_cols),
    (bpnn_single_val_end_df, bpnn_single_val_end_cols),  # Added single BPNN
    (lgb_bpnn_val_end_df, lgb_bpnn_val_end_cols),
    (lgb_bpnn_diff_val_end_df, lgb_bpnn_diff_val_end_cols),
    # 4 models with validation set in the middle
    (lr_val_mid_df, lr_val_mid_cols),
    (bpnn_single_val_mid_df, bpnn_single_val_mid_cols),  # Added single BPNN
    (lgb_bpnn_val_mid_df, lgb_bpnn_val_mid_cols),
    (lgb_bpnn_diff_val_mid_df, lgb_bpnn_diff_val_mid_cols)
]

# 5. Integrate prediction results of 8 models
final_total = data.copy()
all_pred_cols = []

for model_df, model_cols in all_dfs:
    for col in model_cols:
        if col not in final_total.columns:
            final_total[col] = np.nan
            all_pred_cols.append(col)

for window_set in [windows_val_end, windows_val_middle]:
    for window in window_set:
        window_pred_data = window["data"]
        for pred_col in all_pred_cols:
            if pred_col not in window_pred_data.columns:
                continue
            valid_preds = window_pred_data.dropna(subset=[pred_col])
            for _, row in valid_preds.iterrows():
                orig_idx = row["original_index"]
                final_total.loc[final_total["original_index"] == orig_idx, pred_col] = row[pred_col]

# 6. Save final results
final_total_path = os.path.join(save_root, "all_8models_predictions.csv")
final_total.to_csv(final_total_path, index=False)
print(f"\nPrediction results of 8 models saved to: {final_total_path}")
print("\nExplanation of generated prediction columns:")
print("- linear_regression_val_end_return_pred: Linear Regression with validation set at the end (return prediction)")
print("- bpnn_single_val_end_prob_pred: Single BPNN with validation set at the end (up/down probability prediction)")  # Added
print("- lgb_bpnn_fusion_val_end_prob_pred: LightGBM-BPNN Fusion with validation set at the end (up/down probability prediction)")
print("- lgb_bpnn_diffusion_fusion_val_end_prob_pred: LightGBM-BPNN-Diffusion Fusion with validation set at the end (up/down probability prediction)")
print("- linear_regression_val_middle_return_pred: Linear Regression with validation set in the middle (return prediction)")
print("- bpnn_single_val_middle_prob_pred: Single BPNN with validation set in the middle (up/down probability prediction)")  # Added
print("- lgb_bpnn_fusion_val_middle_prob_pred: LightGBM-BPNN Fusion with validation set in the middle (up/down probability prediction)")
print("- lgb_bpnn_diffusion_fusion_val_middle_prob_pred: LightGBM-BPNN-Diffusion Fusion with validation set in the middle (up/down probability prediction)")

cname_list = [
    'linear_regression_val_end_return_pred',
    'lgb_bpnn_fusion_val_end_prob_pred',
    'lgb_bpnn_diffusion_fusion_val_end_prob_pred',
    'linear_regression_val_middle_return_pred',
    'lgb_bpnn_fusion_val_middle_prob_pred',
    'lgb_bpnn_diffusion_fusion_val_middle_prob_pred'
]
col_list = ['#00A1FF', '#5ed935', '#f8ba00', '#ff2501', '#d31876', '#919292']
data.index = pd.to_datetime(data.index, format='%Y%m%d')

mm.plot_if_nav_comparison(data,name_list= cname_list,colors= col_list)
