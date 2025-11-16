# -*- coding: utf-8 -*-
"""
EDA for daily close price (S_DQ_CLOSE)
Dataset: D:/pycharmproject/data/002594_SZ_data.csv
Outputs:
  - ./eda_outputs/summary_stats.csv
  - ./eda_outputs/*.png charts
Notes:
  - Uses matplotlib only (no seaborn)
  - One chart per figure
"""
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Config ----------
DATA_PATH = r"D:/pycharmproject/data/002594_SZ_data.csv"
DATE_COL = "TRADE_DT"
CLOSE_COL = "S_DQ_CLOSE"
CODE_COL = "S_INFO_WINDCODE"  # optional, if present
OUTPUT_DIR = "./eda_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Load & parse ----------
df = pd.read_csv(DATA_PATH, encoding="utf-8")

# Parse date whether it's int yyyymmdd or string
if np.issubdtype(df[DATE_COL].dtype, np.number):
    df[DATE_COL] = pd.to_datetime(df[DATE_COL].astype(str), format="%Y%m%d", errors="coerce")
else:
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

df = df.sort_values(DATE_COL).reset_index(drop=True)

# Keep only necessary columns
cols = [c for c in [CODE_COL, DATE_COL, CLOSE_COL] if c in df.columns]
df = df[cols].copy()

mid = len(df) // 2
df = df.iloc[mid:].copy().reset_index(drop=True)

# ---------- Basic features ----------
df["close"] = pd.to_numeric(df[CLOSE_COL], errors="coerce")
df["log_close"] = np.log(df["close"])
df["ret"] = df["log_close"].diff()            # log return
df["ret_simple"] = df["close"].pct_change()   # simple return

# Rolling stats (20/60/120)
for w in (20, 60, 120):
    df[f"roll_mean_{w}"] = df["close"].rolling(w, min_periods=1).mean()
    df[f"roll_vol_{w}"]  = df["ret"].rolling(w, min_periods=1).std()

# Drawdown
cum_max = df["close"].cummax()
df["drawdown"] = df["close"]/cum_max - 1.0

# ---------- Summary statistics ----------
start_date = df[DATE_COL].min()
end_date   = df[DATE_COL].max()
n_obs      = df["close"].notna().sum()
min_p      = float(df["close"].min())
max_p      = float(df["close"].max())
last_p     = float(df["close"].iloc[-1])

ret = df["ret"].dropna().values
ann_fac = math.sqrt(252.0)
ann_vol = float(np.std(ret, ddof=1) * ann_fac) if len(ret)>1 else float("nan")
ann_drift = float(np.mean(ret) * 252.0) if len(ret)>0 else float("nan")
mdd = float(df["drawdown"].min())

summary = pd.DataFrame({
    "start_date":[start_date],
    "end_date":[end_date],
    "n_obs":[n_obs],
    "price_min":[min_p],
    "price_max":[max_p],
    "price_last":[last_p],
    "ann_vol_logret":[ann_vol],
    "ann_drift_logret":[ann_drift],
    "max_drawdown":[mdd]
})
summary.to_csv(os.path.join(OUTPUT_DIR, "summary_stats.csv"), index=False)

# ---------- Helper: save figure ----------
def savefig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path

# ---------- Plot 1: Close price ----------
plt.figure(figsize=(12,4))
plt.plot(df[DATE_COL], df["close"])
plt.title("Close Price")
plt.xlabel("Date"); plt.ylabel("Price")
plt.grid(True, linestyle="--", alpha=0.5)
p1 = savefig("01_close_price.png")

# ---------- Plot 2: Log price ----------
plt.figure(figsize=(12,4))
plt.plot(df[DATE_COL], df["log_close"])
plt.title("Log Close Price")
plt.xlabel("Date"); plt.ylabel("log(Price)")
plt.grid(True, linestyle="--", alpha=0.5)
p2 = savefig("02_log_close.png")

# ---------- Plot 3: Rolling mean (20/60/120) ----------
plt.figure(figsize=(12,4))
plt.plot(df[DATE_COL], df["close"], linewidth=1.0, label="Close")
plt.plot(df[DATE_COL], df["roll_mean_20"], linewidth=1.0, label="MA20")
plt.plot(df[DATE_COL], df["roll_mean_60"], linewidth=1.0, label="MA60")
plt.plot(df[DATE_COL], df["roll_mean_120"], linewidth=1.0, label="MA120")
plt.legend()
plt.title("Rolling Mean (20/60/120) vs Close")
plt.xlabel("Date"); plt.ylabel("Price")
plt.grid(True, linestyle="--", alpha=0.5)
p3 = savefig("03_rolling_means.png")

# ---------- Plot 4: Rolling volatility of log returns ----------
plt.figure(figsize=(12,4))
plt.plot(df[DATE_COL], df["roll_vol_20"], linewidth=1.0, label="Vol20")
plt.plot(df[DATE_COL], df["roll_vol_60"], linewidth=1.0, label="Vol60")
plt.plot(df[DATE_COL], df["roll_vol_120"], linewidth=1.0, label="Vol120")
plt.legend()
plt.title("Rolling Volatility (log returns)")
plt.xlabel("Date"); plt.ylabel("Std (log return)")
plt.grid(True, linestyle="--", alpha=0.5)
p4 = savefig("04_rolling_volatility.png")

# ---------- Plot 5: Daily log return histogram ----------
plt.figure(figsize=(8,4))
plt.hist(df["ret"].dropna().values, bins=60)
plt.title("Histogram of Daily Log Returns")
plt.xlabel("Log Return"); plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.5)
p5 = savefig("05_return_hist.png")

# ---------- Plot 6: Autocorrelation of log return (up to 30 lags) ----------
def acf(x, nlags=30):
    x = np.asarray(x, float)
    x = x - np.nanmean(x)
    var = np.nanvar(x)
    ac = [1.0]
    for k in range(1, nlags+1):
        if k >= len(x):
            ac.append(np.nan)
            continue
        c = np.nanmean((x[:-k])*(x[k:]))
        ac.append(c/var if var>0 else np.nan)
    return np.array(ac)

acf_vals = acf(df["ret"].dropna().values, nlags=30)
plt.figure(figsize=(10,4))
plt.bar(np.arange(len(acf_vals)), acf_vals, width=0.6)
plt.title("ACF of Daily Log Returns (0..30)")
plt.xlabel("Lag"); plt.ylabel("ACF")
plt.grid(True, linestyle="--", alpha=0.5)
p6 = savefig("06_acf_returns.png")

# ---------- Plot 7: Drawdown curve ----------
plt.figure(figsize=(12,4))
plt.plot(df[DATE_COL], df["drawdown"])
plt.title("Drawdown")
plt.xlabel("Date"); plt.ylabel("Drawdown")
plt.grid(True, linestyle="--", alpha=0.5)
p7 = savefig("07_drawdown.png")

# ---------- Extremes: Top/Bottom daily moves ----------
ret_series = df[[DATE_COL, "ret"]].dropna().copy()
ret_series["abs_ret"] = ret_series["ret"].abs()
top10 = ret_series.nlargest(10, "ret")[[DATE_COL, "ret"]]
bot10 = ret_series.nsmallest(10, "ret")[[DATE_COL, "ret"]]
top10.to_csv(os.path.join(OUTPUT_DIR, "top10_up_days.csv"), index=False)
bot10.to_csv(os.path.join(OUTPUT_DIR, "top10_down_days.csv"), index=False)

print("Saved outputs to:", os.path.abspath(OUTPUT_DIR))
