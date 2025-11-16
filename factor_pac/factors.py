
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import datetime
from statsmodels.regression.linear_model import OLS
import talib


class FactorCalculator:
    def __init__(self, df):
        """初始化因子计算器，传入原始数据DataFrame并保留副本"""
        self.original_df = df.copy()  # 保存原始数据
        self.df = df.copy()  # 用于计算的副本
        # 确保必要列存在（根据因子计算需求）
        required_cols = ['group_volume', 'Close_price', 'Open_price', 
                        'Low_price', 'High_price', 'group_time']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"输入DataFrame缺少必要列：{missing_cols}")


    # 以下因子计算方法与之前完全一致（省略重复代码，实际使用时需保留）
    def volume_spike(self, window=20):
        rolling_mean = self.df['group_volume'].rolling(window).mean().shift(1)
        return self.df['group_volume'] / rolling_mean

    def price_jump_probability(self, threshold=0.001):
        ret = self.df['Close_price'].pct_change()
        return ret.abs().gt(threshold).rolling(30).mean()

    def trade_continuation(self):
        buy_trades = self.df['group_volume'] * (self.df['Close_price'] > self.df['Open_price'])
        sell_trades = self.df['group_volume'] * (self.df['Close_price'] < self.df['Open_price'])
        return (buy_trades - sell_trades).abs() / self.df['group_volume']

    def relative_volatility_index(self, N=5, N1=10, N2=20):
        price_diff = self.df['Close_price'].diff()
        um = np.where(price_diff > 0, self.df['Close_price'].rolling(N1).std(), 0)
        dm = np.where(price_diff < 0, self.df['Close_price'].rolling(N1).std(), 0)
        ua = pd.Series(um).ewm(span=N2, adjust=False).mean().values
        da = pd.Series(dm).ewm(span=N2, adjust=False).mean().values
        rs_high = 100 * ua / (ua + da + 1e-12)
        rs_low = 100 * da / (ua + da + 1e-12)
        return (rs_high + rs_low) / 2

    def MFI(self, window=20):
        tp = (self.df['Close_price'] + self.df['Low_price'] + self.df['High_price']) / 3
        mf = self.df['group_volume'] * tp
        tp_diff = tp.diff()
        pf = np.where(tp_diff > 0, mf, 0)
        nf = np.where(tp_diff < 0, mf, 0)
        pf_sum = pd.Series(pf).rolling(window=window, min_periods=1).sum()
        nf_sum = pd.Series(nf).rolling(window=window, min_periods=1).sum()
        money_ratio = pf_sum / nf_sum
        mfi = 100 - (100 / (1 + money_ratio))
        return np.where(nf_sum == 0, 100, mfi)

    def RSI(self, N=14):
        delta = self.df['Close_price'].ffill()
        return talib.RSI(delta, timeperiod=N)

    def SRSI(self, N=14, N1=14):
        rsi = self.RSI(N)
        min_rsi = pd.Series(rsi).rolling(N1).min()
        max_rsi = pd.Series(rsi).rolling(N1).max()
        denominator = max_rsi - min_rsi
        numerator = rsi - min_rsi
        srs1 = np.where(denominator != 0, numerator / denominator * 100, np.nan)
        srs1 = pd.Series(srs1, index=self.df.index).clip(0, 100)
        return srs1.fillna(0)

    def PVI(self, initial_value=1000):
        volume = self.df['group_volume']
        close = self.df['Close_price']
        pvi = np.zeros(len(self.df))
        pvi[0] = initial_value
        pct_change = close.pct_change()
        volume_prev = volume.shift(1)
        for i in range(1, len(self.df)):
            if volume.iloc[i] > volume_prev.iloc[i]:
                pvi[i] = pvi[i-1] * (1 + pct_change.iloc[i])
            else:
                pvi[i] = pvi[i-1]
        return pd.Series(pvi, index=self.df.index)

    def VHF(self, n=20):
        HCP = self.df['High_price'].rolling(n).max()
        LCP = self.df['Low_price'].rolling(n).min()
        A = abs(HCP - LCP)
        B = self.df['Close_price'].diff().abs().rolling(n).sum()
        return np.where(B != 0, A / B, 0)

    def CVI(self, n=20):
        h_l = self.df['High_price'] - self.df['Low_price']
        ema_hl = h_l.ewm(span=n, adjust=False).mean()
        cvi = (ema_hl - ema_hl.shift(n)) * 100 / ema_hl
        return cvi.fillna(0)

    def CMO(self, n=20):
        delta = self.df['Close_price'].ffill()
        return talib.CMO(delta, timeperiod=n)

    def VIDYA(self, n=20, init_method='sma'):
        close = self.df['Close_price'].ffill().values
        vi = self.CMO(n) / 100
        vi = (vi + 1) / 2
        sc = 2 / (n + 1)
        vidya = np.zeros(len(self.df))
        if init_method == 'sma':
            vidya[:n] = self.df['Close_price'].rolling(n).mean().values[:n]
        else:
            vi = pd.Series(vi).fillna(0.5).values
            vidya[0] = close[0]
        for i in range(n, len(self.df)):
            vidya[i] = sc * vi[i] * close[i] + (1 - sc * vi[i]) * vidya[i-1]
        return pd.Series(vidya, index=self.df.index)

    def UI(self, n=20):
        max_n = self.df['Close_price'].rolling(n).max()
        R = (self.df['Close_price'] - max_n) * 100 / max_n
        sum_sq_R = (R ** 2).rolling(n).sum()
        ui = np.sqrt(sum_sq_R / n)
        return ui.fillna(0)

    def ACD(self, n=20):
        delta = self.df['Close_price'].diff()
        close_prev = self.df['Close_price'].shift(1)
        DIF = np.where(
            delta > 0,
            np.minimum(close_prev, self.df['Low_price']),
            np.maximum(close_prev, self.df['High_price'])
        )
        ACD_s = pd.Series(np.where(delta == 0, 0, DIF), index=self.df.index)
        return ACD_s.rolling(n).sum().fillna(0)

    def TS(self, n=20):
        delta = self.df['Close_price'].diff()
        dif = np.where(delta >= 0, 1, -1)
        dif_s = pd.Series(np.where(delta == 0, 0, dif), index=self.df.index)
        return dif_s.rolling(n).sum()

    def MACD(self, n1=12, n2=26, n3=9):
        close = self.df['Close_price']
        DIF = close.ewm(span=n1, adjust=False).mean() - close.ewm(span=n2, adjust=False).mean()
        DEA = DIF.ewm(span=n3, adjust=False).mean()
        return 2 * (DIF - DEA)

    def VMACD(self, n1=12, n2=26, n3=9):
        volume = self.df['group_volume']
        DIF = volume.ewm(span=n1, adjust=False).mean() - volume.ewm(span=n2, adjust=False).mean()
        DEA = DIF.ewm(span=n3, adjust=False).mean()
        return 2 * (DIF - DEA)

    def TMACD(self, n1=12, n2=26, n3=9):
        group_time = self.df['group_time']
        DIF = group_time.ewm(span=n1, adjust=False).mean() - group_time.ewm(span=n2, adjust=False).mean()
        DEA = DIF.ewm(span=n3, adjust=False).mean()
        return 2 * (DIF - DEA)

    def QST(self, n=20):
        c_l = self.df['Close_price'] - self.df['Open_price']
        return c_l.rolling(n).sum() / n

    def DDI(self, ddi_window=20):
        high = self.df['High_price']
        low = self.df['Low_price']
        condition_dmz = (high + low) <= (high.shift(1) + low.shift(1))
        dmz = np.where(
            condition_dmz,
            0,
            np.maximum(np.abs(high - high.shift(1)), np.abs(low - low.shift(1)))
        )
        condition_dmf = (high + low) > (high.shift(1) + low.shift(1))
        dmf = np.where(
            condition_dmf,
            0,
            np.maximum(np.abs(high - high.shift(1)), np.abs(low - low.shift(1)))
        )
        sum_dmz = pd.Series(dmz).rolling(ddi_window).sum()
        sum_dmf = pd.Series(dmf).rolling(ddi_window).sum()
        denominator = sum_dmz + sum_dmf
        diz = np.where(denominator == 0, 0, sum_dmz / denominator * 100)
        dif = np.where(denominator == 0, 0, sum_dmf / denominator * 100)
        return diz - dif

    def KVO(self, signal_window=13, kvo_window_s=34, kvo_window_l=55):
        high = self.df['High_price'].values
        low = self.df['Low_price'].values
        close = self.df['Close_price'].values
        volume = self.df['group_volume'].values
        size = len(close)
        high_shift1 = pd.Series(high).shift(1).fillna(np.nan).values
        low_shift1 = pd.Series(low).shift(1).fillna(np.nan).values
        close_shift1 = pd.Series(close).shift(1).fillna(np.nan).values
        condition_tr = (high + low + close) > (high_shift1 + low_shift1 + close_shift1)
        tr = np.where(condition_tr, 1, -1)
        dm = high - low
        cm = np.zeros(size)
        cm[0] = dm[0]
        for i in range(1, size):
            if tr[i] == tr[i-1]:
                cm[i] = cm[i-1] + dm[i]
            else:
                cm[i] = dm[i-1] + dm[i]
        ratio = np.abs(2 * (dm / (cm + 1e-12) - 1))
        ratio = pd.Series(ratio).fillna(0).values
        vf = volume * ratio * tr * 100
        kvo = pd.Series(vf).ewm(span=kvo_window_s, adjust=False).mean().values - \
              pd.Series(vf).ewm(span=kvo_window_l, adjust=False).mean().values
        kvo_signal = pd.Series(kvo).ewm(span=signal_window, adjust=False).mean().values
        return kvo, kvo_signal


    def get_combined_df(self):
        """生成包含原始数据和所有因子的合并DataFrame"""
        # 计算所有因子
        factors = pd.DataFrame(index=self.df.index)
        factors['volume_spike'] = self.volume_spike()
        factors['price_jump_probability'] = self.price_jump_probability()
        factors['trade_continuation'] = self.trade_continuation()
        factors['relative_volatility_index'] = self.relative_volatility_index()
        factors['MFI'] = self.MFI()
        factors['RSI'] = self.RSI()
        factors['SRSI'] = self.SRSI()
        factors['PVI'] = self.PVI()
        factors['VHF'] = self.VHF()
        factors['CVI'] = self.CVI()
        factors['CMO'] = self.CMO()
        factors['VIDYA'] = self.VIDYA()
        factors['UI'] = self.UI()
        factors['ACD'] = self.ACD()
        factors['TS'] = self.TS()
        factors['MACD'] = self.MACD()
        factors['VMACD'] = self.VMACD()
        factors['TMACD'] = self.TMACD()
        factors['QST'] = self.QST()
        factors['DDI'] = self.DDI()
        kvo, kvo_signal = self.KVO()
        factors['KVO'] = kvo
        factors['KVO_signal'] = kvo_signal
        
        # 将原始数据与因子合并（使用索引对齐）
        combined_df = self.original_df.join(factors, how='left')
        return combined_df

def factor_gen(dict):
    result_dfs = []
    for date_key, df in dict.items():
        if is_datetime64_any_dtype(df.index):
            df = df.reset_index()
        if 'date' not in df.columns:
            df.rename(columns = {'Last_timestamp':'date'},inplace = True)
        if 'datetime' in  df.columns:
            df.drop(columns =['datetime'],inplace = True)
        df['date'] = df['date'].dt.tz_localize(None)
        result_dfs.append(df)
    result_combined = pd.concat(result_dfs, axis=0, ignore_index=True)
    # result_combined.drop(columns =['date'],inplace = True)
    # result_combined.rename(columns = {'datetime':'date'},inplace = True)
    result_combined.set_index('date',inplace = True)
    return result_combined