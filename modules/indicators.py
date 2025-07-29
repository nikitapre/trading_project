# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python (trading_env)
#     language: python
#     name: trading_env
# ---

# %% [markdown]
# # Импорты

# %%
import pandas as pd
import pandas_ta as ta
import datetime as dt
import time
import math
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, minmax_scale
from datetime import datetime
from scipy.signal import argrelextrema
from scipy.special import expit
from scipy.stats import zscore
from scipy.stats import linregress
from scipy.fft import rfft, rfftfreq
from sklearn.linear_model import Ridge
from scipy.stats import wasserstein_distance  # Альтернатива, если нет POT
import ot
from numba import njit
import swifter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from pytz import timezone
from scipy.signal import savgol_filter
from numba import jit
from scipy.signal import welch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LinearRegression
from pandas_ta.trend import psar
from pandas_ta.volatility import atr
from pandas_ta.momentum import rsi


# %% [markdown]
# # Функция расчета RSI

# %%
# RSI вручную
def compute_rsi(series, period=14, eps=1e-8):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / (avg_loss + eps)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# %% [markdown]
# RSI levels

# %%
def add_rsi_level_signal(
    df: pd.DataFrame,
    rsi_period: int = 14,
    lookback_window: int = 200,
    tolerance: float = 0.03,
    min_touch_count: int = 2,
) -> pd.DataFrame:
    """
    Находит уровни RSI с помощью кластеризации экстремумов и добавляет бинарный признак:
    - `rsi_near_level = 1`, если текущий RSI близко к одному из уровней.

    Параметры:
        df: DataFrame с ценами (должен содержать столбец 'Close').
        rsi_period: период для расчета RSI (по умолчанию 14).
        lookback_window: глубина анализа (по умолчанию 200 свечей).
        tolerance: допустимое отклонение уровня (относительное, 0.03 = 3%).
        min_touch_count: минимальное количество касаний для формирования уровня.

    Возвращает:
        DataFrame с добавленным столбцом `rsi_near_level` (1 только для последней свечи).
    """
    if df.empty:
        raise ValueError("DataFrame пуст!")

    df = df.copy()
    lookback_window = min(lookback_window, len(df))

    # 1. Вычисляем RSI (используем pandas_ta.rsi)
    df["rsi"] = ta.rsi(df["Close"], length=rsi_period)  # <-- Исправлено здесь

    # 2. Берем последние `lookback_window` значений RSI (игнорируем NaN)
    recent_rsi = df["rsi"].dropna().iloc[-lookback_window:]
    if len(recent_rsi) < 2:  # Недостаточно данных для анализа
        df["rsi_near_level"] = 0
        return df

    # 3. Находим локальные экстремумы RSI (максимумы и минимумы)
    rsi_values = recent_rsi.values
    extrema = []
    for i in range(1, len(rsi_values) - 1):
        if (rsi_values[i] > rsi_values[i - 1]) and (rsi_values[i] > rsi_values[i + 1]):
            extrema.append(rsi_values[i])  # Локальный максимум
        elif (rsi_values[i] < rsi_values[i - 1]) and (rsi_values[i] < rsi_values[i + 1]):
            extrema.append(rsi_values[i])  # Локальный минимум

    if not extrema:  # Нет экстремумов → нет уровней
        df["rsi_near_level"] = 0
        return df

    # 4. Кластеризация DBSCAN (группировка уровней)
    X = np.array(extrema).reshape(-1, 1)
    eps_abs = np.mean(X) * tolerance  # Переводим относительный tolerance в абсолютный
    db = DBSCAN(eps=eps_abs, min_samples=min_touch_count).fit(X)

    # 5. Извлекаем уровни (игнорируем шум `label=-1`)
    levels = [
        np.mean(X[db.labels_ == lbl])
        for lbl in set(db.labels_)
        if lbl != -1
    ]

    # 6. Проверяем, находится ли текущий RSI рядом с каким-то уровнем
    current_rsi = df["rsi"].iloc[-1]
    close_to_level = any(
        abs(current_rsi - level) / level <= tolerance
        for level in levels
    )

    df["rsi_near_level"] = 0
    if close_to_level:
        df.at[df.index[-1], "rsi_near_level"] = 1

    return df


# %% [markdown]
# Угл наклона RSI

# %%
def calculate_rsi_slope(df, close_col='Close', length=50, lookback=10, slope_coeff=10):
    """
    Рассчитывает RSI и его нормализованный наклон за lookback период
    
    Параметры:
        df - DataFrame с ценовыми данными
        close_col - название столбца с ценами закрытия
        length - период для RSI
        lookback - окно для расчета угла наклона
        slope_coeff - коэффициент усиления наклона перед tanh
        
    Возвращает:
        Series с нормализованными значениями наклона RSI
    """
    # Рассчитываем RSI
    rsi = ta.rsi(df[close_col], length=length)
    
    # Инициализируем массив для наклонов
    slopes = np.zeros(len(df))
    slopes[:] = np.nan  # Первые значения будут NaN
    
    # Рассчитываем наклон для каждого окна
    for i in range(lookback, len(df)):
        y = rsi.iloc[i-lookback:i].values
        x = np.arange(len(y))
        
        # Игнорируем окна с пропусками
        if np.isnan(y).any():
            continue
            
        slope = np.polyfit(x, y, 1)[0]  # Линейный коэффициент
        slopes[i] = np.tanh(slope * slope_coeff)  # Нормализация
        
    return slopes


# %% [markdown]
# **CMF**

# %%
def compute_cmf(df, length=20, eps=1e-8):
    mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + eps)
    mfv *= df['Volume']
    cmf = mfv.rolling(window=length).sum() / (df['Volume'].rolling(window=length).sum() + eps)
    return cmf


# %% [markdown]
# **Пробой уровня сопротивления**

# %%
def add_breakout_flags(df, resistance_window=20, lookback=5):
    df = df.copy()

    # Сопротивление = максимум из прошлых resistance_window свечей
    df['resistance'] = df['High'].rolling(window=resistance_window, min_periods=1).max().shift(1)

    # Текущий пробой (High > resistance)
    df['breakout_now'] = (df['High'] > df['resistance']).astype(int)

    # Проверяем пробой в последние lookback свечей (включая текущую)
    df['breakout_in_5'] = 0
    for i in range(lookback + 1):  # +1, потому что range(5) даёт 0,1,2,3,4 (5 значений)
        df['breakout_in_5'] |= (df['High'].shift(i) > df['resistance'].shift(i)).astype(int)

    # Удаляем resistance (если не нужно для отладки)
    df.drop(columns=['resistance'], inplace=True)

    return df


# %% [markdown]
# **Пробой уровня поддержки**

# %%
def add_breakdown_flags(df, support_window=20, lookback=5):
    df = df.copy()

    # Расчёт уровня поддержки (минимум за support_window свечей, сдвинутый на 1 вперёд)
    df['support'] = df['Low'].rolling(window=support_window, min_periods=1).min().shift(1)

    # Текущий пробой вниз (Low < support)
    df['breakdown_now'] = (df['Low'] < df['support']).astype(int)

    # Проверяем пробой вниз в последние lookback свечей (включая текущую)
    df['breakdown_in_5'] = 0
    for i in range(lookback + 1):  # +1 чтобы включить текущую свечу (i=0)
        df['breakdown_in_5'] |= (df['Low'].shift(i) < df['support'].shift(i)).astype(int)

    # Удаляем промежуточные столбцы (можно закомментировать для отладки)
    df.drop(columns=['support'], inplace=True)

    return df


# %% [markdown]
# # Уровни фибоначчи

# %%
def get_fibonacci_flags(df, window=300, threshold=0.003, debug=False):
    try:
        if df is None or df.empty or len(df) < window:
            return pd.DataFrame(index=df.index)

        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        level_cols = [f'fib_dist_{int(level*1000)}' for level in fib_levels]
        result = pd.DataFrame(np.nan, index=df.index, columns=level_cols)

        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values

        rolling_high = pd.Series(highs).rolling(window, min_periods=window).max().values
        rolling_low = pd.Series(lows).rolling(window, min_periods=window).min().values
        price_ranges = rolling_high - rolling_low
        rel_changes = price_ranges / (rolling_low + 1e-10)

        for i in range(window, len(df)):
            if rel_changes[i] < threshold or np.isnan(price_ranges[i]):
                continue

            high = rolling_high[i]
            low = rolling_low[i]
            price_range = price_ranges[i]

            window_highs = highs[i-window:i]
            window_lows = lows[i-window:i]
            if len(window_highs) == 0 or len(window_lows) == 0:
                continue

            high_pos = np.argmax(window_highs)
            low_pos = np.argmin(window_lows)
            is_uptrend = high_pos > low_pos
            current_price = closes[i]

            if is_uptrend:
                fib_values = high - price_range * np.array(fib_levels)
            else:
                fib_values = low + price_range * np.array(fib_levels)

            dists = (current_price - fib_values) / (current_price + 1e-10)
            result.iloc[i] = dists.astype(np.float32)

        return result

    except Exception as e:
        if debug:
            print(f"⚠️ Fibonacci error: {str(e)}")
        return pd.DataFrame(index=df.index)


# %% [markdown]
# Фиба 0,5
#

# %%
def fib_dist_050_last50(df: pd.DataFrame,
                        window: int = 50,
                        threshold: float = 0.03,
                        debug: bool = False) -> pd.Series:
    """
    Вычисляет расстояние от текущей цены до уровня 0.5 по последней сетке Фибоначчи
    (если был рост/падение более threshold за последние window свечей).
    
    Возвращает Series со значениями от -1 до 1, -1 если уровень не построен.
    """
    try:
        if df is None or df.empty or len(df) < window:
            return pd.Series(-1, index=df.index, name='fib_dist_050_last50')

        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        result = np.full(len(df), -1.0, dtype=np.float32)  # по умолчанию -1

        for i in range(window, len(df)):
            high = np.max(highs[i - window:i])
            low = np.min(lows[i - window:i])
            current_price = closes[i]
            price_range = high - low
            if low == 0 or price_range / low < threshold:
                continue

            high_pos = np.argmax(highs[i - window:i])
            low_pos = np.argmin(lows[i - window:i])
            is_uptrend = high_pos > low_pos  # рост

            # Уровень 0.5 по тренду
            fib_level = high - 0.5 * price_range if is_uptrend else low + 0.5 * price_range

            # Расстояние до уровня 0.5, нормализуем
            dist = (current_price - fib_level) / price_range
            dist = max(-1, min(1, dist))  # ограничим в пределах [-1, 1]

            result[i] = dist

        return pd.Series(result, index=df.index, name='fib_dist_050_last50')

    except Exception as e:
        if debug:
            print(f"⚠️ Fibonacci error: {e}")
        return pd.Series(-1, index=df.index, name='fib_dist_050_last50')


# %% [markdown]
# # Cтохастик RSI

# %%
def calc_stochastic_rsi(df, rsi_period=14, stoch_period=14, smooth_k=3, smooth_d=3):

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Stochastic RSI
    min_rsi = rsi.rolling(window=stoch_period).min()
    max_rsi = rsi.rolling(window=stoch_period).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi + 1e-10)

    # Smoothed %K and %D
    k = stoch_rsi.rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()

    df['stoch_rsi_k'] = k
    df['stoch_rsi_d'] = d

    # Бычье пересечение: %K пересекает %D снизу вверх
    bull_cross = (k.shift(1) < d.shift(1)) & (k >= d)
    bull_cross_indices = df.index[bull_cross.fillna(False)]

    # Создаём колонку для бычьего пересечения
    stoch_cross_bull = pd.Series(0, index=df.index)

    for idx in bull_cross_indices:
        start = df.index.get_loc(idx)
        stoch_cross_bull.iloc[start:start + 4] = 1  # текущий бар + 3 после него

    df['stoch_cross_bull'] = stoch_cross_bull.clip(upper=1).fillna(0)

    return df


# %% [markdown]
# # RSI дивергенции с ценой

# %%
def find_bullish_rsi_divergence(df, rsi_column='RSI7', lookback=5):
    """
    Находит бычьи дивергенции RSI-цена (классическая, скрытая, расширенная)
    
    Возвращает:
        - 'has_divergence_<lookback>'
        - 'divergence_power_<lookback>': нормализованная сила дивергенции (-1..1)
    """
    price_col = f'price_low_{lookback}'
    rsi_col = f'rsi_low_{lookback}'
    signal_col = f'has_divergence_{lookback}'
    power_col = f'divergence_power_{lookback}'

    # Локальные минимумы
    df[price_col] = df['Low'].rolling(lookback, center=True).min() == df['Low']
    df[rsi_col] = df[rsi_column].rolling(lookback, center=True).min() == df[rsi_column]
    
    # Инициализация
    df[signal_col] = 0
    df[power_col] = 0.0

    for i in range(lookback * 2, len(df)):
        if df[price_col].iloc[i] and df[rsi_col].iloc[i]:
            for j in range(i-1, max(-1, i - lookback*2), -1):
                if df[price_col].iloc[j]:
                    current_low = df['Low'].iloc[i]
                    prev_low = df['Low'].iloc[j]
                    current_rsi = df[rsi_column].iloc[i]
                    prev_rsi = df[rsi_column].iloc[j]
                    rsi_diff = current_rsi - prev_rsi
                    
                    # Диапазон RSI в окне
                    rsi_window = df[rsi_column].iloc[j:i+1]
                    rsi_range = rsi_window.max() - rsi_window.min()
                    if rsi_range == 0:
                        normalized_power = 0
                    else:
                        normalized_power = round(rsi_diff / rsi_range, 3)

                    # Проверка типов дивергенций
                    if ((current_low < prev_low and current_rsi > prev_rsi) or    # Классическая
                        (current_low > prev_low and current_rsi < prev_rsi) or    # Скрытая
                        (abs(current_low - prev_low) < 0.01 * prev_low and current_rsi > prev_rsi)):  # Расширенная
                        
                        df.loc[df.index[i], signal_col] = 1
                        df.loc[df.index[i], power_col] = normalized_power
                    break

    return df


# %%
def rsi_divergence(df, length=14, lbL=5, lbR=5, rangeLower=5, rangeUpper=60, signal_duration=5):
    """
    Векторизованная и пригодная для реального времени версия RSI-дивергенций.
    Сигналы выставляются на текущей свече и нескольких следующих.
    """
    df = df.copy()
    df['rsi'] = df.ta.rsi(close='Close', length=length)
    df['bull_signal'] = 0
    df['hidden_bull_signal'] = 0

    lows = df['Low'].values
    rsi = df['rsi'].values
    n = len(df)

    for i in range(lbL + lbR, n):
        pivot_idx = i - lbR

        # Проверка текущего пивота (на момент i)
        is_low_pivot = (
            lows[pivot_idx] == np.min(lows[pivot_idx - lbL:pivot_idx + lbR + 1]) and
            rsi[pivot_idx] == np.min(rsi[pivot_idx - lbL:pivot_idx + lbR + 1])
        )

        if not is_low_pivot:
            continue

        # Ищем предыдущий пивот
        for j in range(i - lbR - 1, lbL + lbR - 1, -1):
            prev_pivot_idx = j - lbR
            if prev_pivot_idx - lbL < 0 or prev_pivot_idx + lbR >= n:
                continue

            prev_pivot = (
                lows[prev_pivot_idx] == np.min(lows[prev_pivot_idx - lbL:prev_pivot_idx + lbR + 1]) and
                rsi[prev_pivot_idx] == np.min(rsi[prev_pivot_idx - lbL:prev_pivot_idx + lbR + 1])
            )

            if not prev_pivot:
                continue

            # Классическая бычья дивергенция
            price_cond = lows[pivot_idx] < lows[prev_pivot_idx]
            rsi_cond = rsi[pivot_idx] > rsi[prev_pivot_idx]

            if price_cond and rsi_cond:
                for k in range(signal_duration):
                    if i + k < n:
                        df.at[i + k, 'bull_signal'] = 1

            # Скрытая бычья дивергенция
            hidden_price_cond = lows[pivot_idx] > lows[prev_pivot_idx]
            hidden_rsi_cond = rsi[pivot_idx] < rsi[prev_pivot_idx]

            if hidden_price_cond and hidden_rsi_cond:
                for k in range(signal_duration):
                    if i + k < n:
                        df.at[i + k, 'hidden_bull_signal'] = 1

            break  # используем только первый найденный предыдущий пивот

    return df


# %% [markdown]
# **Профиль объема**

# %%
def add_normalized_volume_profile(df, window=20, lag=0, change_window=5, sma_window=10):
    """
    Добавляет в DataFrame ТОЛЬКО нормализованные значения:
    - Нормализованный профиль объёмов (VWAP).
    - Нормализованный профиль с лагом (если lag > 0).
    - Нормализованную скорость изменения профиля.
    - Нормализованную SMA профиля.
    """
    df = df.copy()
    
    # 1. Расчёт VWAP (профиль объёмов)
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP_Price'] = (df['Typical_Price'] * df['Volume']).rolling(window).sum() / df['Volume'].rolling(window).sum()
    
    # 2. Расчёт SMA профиля
    df['VP_SMA'] = df['VP_Price'].rolling(sma_window).mean()
    
    # 3. Нормализация VP_Price
    price_mean = df['VP_Price'].rolling(sma_window).mean()
    price_std = df['VP_Price'].rolling(sma_window).std()
    df['VP_Norm'] = (df['VP_Price'] - price_mean) / price_std
    
    # 4. Нормализация SMA (используем ОТДЕЛЬНЫЕ среднее и отклонение для SMA)
    sma_mean = df['VP_SMA'].rolling(sma_window).mean()
    sma_std = df['VP_SMA'].rolling(sma_window).std()
    df['VP_SMA_Norm'] = (df['VP_SMA'] - sma_mean) / sma_std
    
    # 5. Нормализация изменения профиля
    change_mean = df['VP_Price'].pct_change(change_window).rolling(sma_window).mean()
    change_std = df['VP_Price'].pct_change(change_window).rolling(sma_window).std()
    df['VP_Change_Norm'] = (df['VP_Price'].pct_change(change_window) - change_mean) / change_std
    
    # 6. Добавление лага (если требуется)
    if lag > 0:
        df['VP_Norm_Lag'] = df['VP_Norm'].shift(lag)
    
    # Удаляем все ненормализованные колонки
    cols_to_drop = ['Typical_Price', 'VP_Price', 'VP_SMA']
    df.drop(cols_to_drop, axis=1, inplace=True, errors='ignore')
    
    return df


# %% [markdown]
# **Фильтр объема**

# %%
def add_volume_filter(df: pd.DataFrame, 
                      use_ema: bool = False,
                      window_short: int = 20,
                      window_long: int = 100) -> pd.DataFrame:
    """
    Добавляет индикатор Volume Filter с защитой от inf/nan и возможностью использовать EMA.
    
    Parameters:
        df (pd.DataFrame): DataFrame с колонками ['High','Low','Open','Close','Volume']
        use_ema (bool): Если True, использует EMA вместо SMA (меньше лага)
        window_short (int): Период короткого скользящего среднего (по умолчанию 20)
        window_long (int): Период длинного скользящего среднего (по умолчанию 100)
    
    Returns:
        pd.DataFrame: Исходный DataFrame с добавленной колонкой 'volume_filter'
    """
    
    # Проверка валидности окон
    if window_short >= window_long:
        raise ValueError("Короткое окно должно быть меньше длинного")
    
    # Копируем DataFrame чтобы избежать предупреждений
    df = df.copy()
    
    try:
        if use_ema:
            # Версия с экспоненциальным сглаживанием
            short_ma = df['Volume'].ewm(span=window_short, min_periods=1).mean()
            long_ma = df['Volume'].ewm(span=window_long, min_periods=1).mean()
        else:
            # Стандартная версия с простым скользящим средним
            short_ma = df['Volume'].rolling(window=window_short, min_periods=1).mean()
            long_ma = df['Volume'].rolling(window=window_long, min_periods=1).mean()
        
        # Вычисляем процентное отклонение с защитой от деления на 0
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(long_ma != 0, short_ma / long_ma, 1)  # При делении на 0 возвращаем 1 (0% изменения)
            volume_filter = (ratio - 1) * 100
            
            # Заменяем inf/nan на 0 (кроме начальных значений, где это нормально)
            mask = (long_ma == 0) | (~np.isfinite(volume_filter))
            volume_filter = np.where(mask, 0, volume_filter)
        
        df['volume_filter'] = volume_filter
        
    except Exception as e:
        raise RuntimeError(f"Ошибка при расчете Volume Filter: {str(e)}")
    
    return df

# Пример использования:
# df = add_volume_filter(df)  # SMA версия по умолчанию
# df = add_volume_filter(df, use_ema=True)  # EMA версия
# df = add_volume_filter(df, window_short=14, window_long=50)  # С кастомными периодами


# %% [markdown]
# # ATR

# %%
def add_atr_normalized(
    df: pd.DataFrame,
    windows: list = [14, 24, 48],
    ema: bool = True,
    normalization_window: int = 100,
    add_velocity: bool = True,
    add_acceleration: bool = True,
    velocity_window: int = 5,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Улучшенная нормализация ATR в диапазон [-1, 1] через скользящие квантили.
    Подходит для ML и разных криптопар.
    
    Parameters:
        df: DataFrame с колонками ['High', 'Low', 'Close']
        windows: Периоды ATR
        ema: Использовать EMA вместо SMA
        normalization_window: Окно для расчета квантилей
        add_velocity: Добавить скорость изменения
        add_acceleration: Добавить ускорение
        velocity_window: Шаг для производных
        eps: Малая константа для избежания деления на 0
    """
    df = df.copy()
    
    # Расчёт True Range
    prev_close = df['Close'].shift(1)
    tr = pd.DataFrame({
        'HL': df['High'] - df['Low'],
        'HC': abs(df['High'] - prev_close),
        'LC': abs(df['Low'] - prev_close)
    }).max(axis=1)

    for window in windows:
        # Расчёт ATR
        atr = tr.ewm(span=window, min_periods=1).mean() if ema else tr.rolling(window=window, min_periods=1).mean()
        col_name = f'atr_{window}_norm'
        
        # Нормализация в [-1, 1] через скользящие квантили
        rolling_min = atr.rolling(normalization_window, min_periods=1).min()
        rolling_max = atr.rolling(normalization_window, min_periods=1).max()
        range_ = rolling_max - rolling_min + eps
        
        df[col_name] = 2 * ((atr - rolling_min) / range_) - 1  # Формула для [-1, 1]
        
        # Производные
        if add_velocity:
            df[f'{col_name}_velocity'] = df[col_name].diff(velocity_window) / velocity_window
        
        if add_acceleration and add_velocity:
            df[f'{col_name}_acceleration'] = df[f'{col_name}_velocity'].diff(velocity_window)

    return df


# %%
def add_atr_normalized_rsi_weight(df: pd.DataFrame, 
                    windows: list = [14, 24, 48],
                    ema: bool = True,
                    normalize: bool = True,
                    add_velocity: bool = True,
                    add_acceleration: bool = True,
                    rsi_period: int = 14,
                    slope_window: int = 5) -> pd.DataFrame:
    """
    Добавляет несколько ATR с разными окнами и их производные, включая RSI-взвешенные версии.
    
    Parameters:
        df (pd.DataFrame): DataFrame с колонками ['High', 'Low', 'Close']
        windows (list): Список периодов ATR (по умолчанию [14, 24, 48])
        ema (bool): Использовать EMA (True) или SMA (False) 
        normalize (bool): Если True, возвращает ATR в % от цены закрытия и добавляет RSI-взвешенные колонки
        add_velocity (bool): Добавлять скорость изменения ATR
        add_acceleration (bool): Добавлять ускорение изменения ATR
        rsi_period (int): Период для расчета RSI (по умолчанию 14)
        slope_window (int): Окно для расчета угла наклона RSI (по умолчанию 5)
    
    Returns:
        pd.DataFrame: С добавленными колонками ATR и производными для каждого окна
    """
    eps=1e-8
    df = df.copy()
    
    # Вычисляем RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Рассчитываем угол наклона RSI за последние slope_window свечей
    rsi_slope = rsi.diff(slope_window) / slope_window
    
    # Упрощенный расчет веса: RSI * угол наклона RSI
    rsi_weight = rsi / (rsi_slope+ eps)
    
    for window in windows:
        # Расчет True Range (TR)
        prev_close = df['Close'].shift(1)
        tr = pd.DataFrame({
            'HL': df['High'] - df['Low'],
            'HC': abs(df['High'] - prev_close),
            'LC': abs(df['Low'] - prev_close)
        }).max(axis=1)
        
        # Расчет ATR (EMA или SMA)
        atr = tr.ewm(span=window, min_periods=1).mean() if ema else tr.rolling(window=window, min_periods=1).mean()
        
        # Базовое имя колонки
        base_col = f'atr_{window}'
        
        # Нормализация (если включена)
        if normalize:
            norm_col = f'{base_col}_norm'
            df[norm_col] = (atr / df['Close']) * 100  # ATR в % от цены закрытия
            df[norm_col] = df[norm_col].replace([np.inf, -np.inf], np.nan)
            
            # Добавляем RSI-взвешенную версию
            df[f'{norm_col}_rsi_weight'] = df[norm_col] * rsi_weight
            
            # Работаем с нормализованным ATR для производных
            working_col = norm_col
        else:
            # Если нормализация отключена, работаем с абсолютным ATR
            df[base_col] = atr
            working_col = base_col
        
        # Добавляем velocity (скорость изменения)
        if add_velocity:
            velocity_col = f'{working_col}_velocity'
            df[velocity_col] = df[working_col].diff() / df[working_col].shift(1)
        
        # Добавляем acceleration (ускорение изменения)
        if add_acceleration and add_velocity:
            acceleration_col = f'{working_col}_acceleration'
            df[acceleration_col] = df[velocity_col].diff()
    
    return df


# %% [markdown]
# Угол наклона ATR

# %%
def calculate_atr_slope(df, length=14, lookback=5, slope_coeff=5):
    """
    Рассчитывает нормализованный наклон ATR за lookback период
    
    Parameters:
        df - DataFrame с ценовыми данными
        length - период для ATR
        lookback - окно для расчета угла наклона
        slope_coeff - коэффициент усиления наклона перед tanh
        
    Returns:
        Series с нормализованными значениями наклона ATR
    """
    # Сначала добавляем ATR с нужным периодом
    df = add_atr_normalized(df, windows=[length], normalize=False, 
                           add_velocity=False, add_acceleration=False)
    
    atr_col = f'atr_{length}'
    slopes = np.zeros(len(df))
    slopes[:] = np.nan
    
    for i in range(lookback, len(df)):
        y = df[atr_col].iloc[i-lookback:i].values
        x = np.arange(len(y))
        
        if np.isnan(y).any():
            continue
            
        slope = np.polyfit(x, y, 1)[0]
        slopes[i] = np.tanh(slope * slope_coeff)
        
    return slopes


# %% [markdown]
# EMA для ATR

# %%
def calculate_atr_ema_normalized(df, length_atr=14, length_ema=20, eps=1e-5):
    """
    Рассчитывает нормализованное расстояние между EMA от ATR и текущим ATR
    
    Parameters:
        df - DataFrame с ценовыми данными
        length_atr - период для ATR
        length_ema - период для EMA
        eps - малое число для избежания деления на 0
        
    Returns:
        Series с нормализованными значениями расстояния в диапазоне [-1, 1]
    """
    # Сначала добавляем ATR с нужным периодом
    df = add_atr_normalized(df, windows=[length_atr], normalize=False,
                          add_velocity=False, add_acceleration=False)
    
    atr_col = f'atr_{length_atr}'
    atr = df[atr_col]
    atr_ema = ta.ema(atr, length=length_ema)
    
    # Нормализация расстояния между EMA(ATR) и текущим ATR
    normalized_diff = np.clip((atr_ema - atr) / (atr + eps), -1, 1)
    
    return normalized_diff


# Пример использования: df['ATR_50_EMA_20_Norm'] = calculate_atr_ema_normalized(df, length_atr=50, length_ema=20)

# %% [markdown]
# atr_24_norm_dynamics

# %%
def add_atr_24_norm_dynamics(df, col='atr_24_norm'):
    """
    Добавляет производные признаки от atr_24_norm:
    - Скользящее стандартное отклонение
    - Скользящее среднее изменения
    - Процентное изменение
    - Скользящее среднее процентного изменения
    - Скользящая производная (разность) первого порядка
    """
    df = df.copy()

    # Скользящее стандартное отклонение за 10 и 30 баров
    df[f'{col}_rolling_std_10'] = df[col].rolling(window=10).std()
    df[f'{col}_rolling_std_30'] = df[col].rolling(window=30).std()

    # Разность первого порядка
    df[f'{col}_diff_1'] = df[col].diff()
    df[f'{col}_rolling_diff_mean_10'] = df[col].diff().rolling(window=10).mean()

    # Процентное изменение
    df[f'{col}_pct_change'] = df[col].pct_change()
    df[f'{col}_pct_change_rolling_mean_10'] = df[col].pct_change().rolling(window=10).mean()

    # Производная второго порядка (ускорение)
    df[f'{col}_diff2'] = df[col].diff().diff()

    return df


# %% [markdown]
# доп варианты ATR

# %%
def atr_local_extremes(df, atr_col='atr_24_norm', window=14):
    """Детектирует точки перегиба волатильности"""
    df[f'{atr_col}_low'] = df[atr_col].rolling(window).min() == df[atr_col]
    df[f'{atr_col}_high'] = df[atr_col].rolling(window).max() == df[atr_col]
    return df



# %% [markdown]
# Еще доп варианты ATR

# %%
def add_atr_features(df: pd.DataFrame, 
                     atr_col: str = 'atr_24_norm', 
                     compress_window: int = 5,
                     expansion_threshold: int = 2) -> pd.DataFrame:
    """
    Добавляет признаки сжатия и расширения волатильности на основе ATR
    
    Параметры:
    - atr_col: колонка с нормализованным ATR
    - compress_window: окно для определения среднего уровня сжатия
    - expansion_threshold: пороговое количество периодов для определения расширения
    """
    df = df.copy()
    
    # Признак сжатия волатильности
    df[f'{atr_col}_compress'] = df[atr_col] < df[atr_col].rolling(compress_window).mean()
    
    # Признак расширения волатильности
    df[f'{atr_col}_expansion'] = (df[atr_col] > df[atr_col].shift(expansion_threshold)) & df[f'{atr_col}_compress'].shift(1)
    
    return df


def add_atr_regimes(df: pd.DataFrame, 
                    atr_col: str = 'atr_24_norm',
                    regime_window: int = 50,
                    low_multiplier: float = 0.5,
                    high_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Добавляет колонку с режимами волатильности на основе ATR
    
    Параметры:
    - atr_col: колонка с нормализованным ATR
    - regime_window: окно для определения среднего уровня волатильности
    - low_multiplier: порог для низкой волатильности
    - high_multiplier: порог для высокой волатильности
    """
    df = df.copy()
    
    # Вычисление среднего ATR
    atr_mean = df[atr_col].rolling(regime_window).mean()
    
    # Определение режимов
    conditions = [
        df[atr_col] < (atr_mean * low_multiplier),  # низкая волатильность
        df[atr_col] > (atr_mean * high_multiplier)   # высокая волатильность
    ]
    
    choices = [-1, 1]  # -1 = low, 0 = normal, 1 = high
    
    # Создание колонки с режимами
    df[f'{atr_col}_regime'] = np.select(conditions, choices, default=0)
    
    return df


def add_price_atr_interaction(df: pd.DataFrame,
                             atr_col: str = 'atr_24_norm',
                             amplitude_window: int = 3,
                             ratio_threshold: float = 0.8,  # Более мягкий порог
                             use_regime: bool = False):     # Опциональное использование режима
    """
    Улучшенная версия с настраиваемыми параметрами
    """
    df = df.copy()
    
    candle_amplitude = df['High'] - df['Low']
    df['candle_atr_ratio'] = candle_amplitude / (df[atr_col] * df['Close'] / 100)
    df['candle_atr_ratio_ma'] = df['candle_atr_ratio'].rolling(amplitude_window).mean()
    
    # Базовые условия
    conditions = [
        (df['Close'] > df['Open']),
        (df['candle_atr_ratio'] > ratio_threshold),
        (df[atr_col] < df[atr_col].shift(1))
    ]
    
    # Добавляем условие по режиму только если нужно
    if use_regime and f'{atr_col}_regime' in df.columns:
        conditions.append(df[f'{atr_col}_regime'] == -1)
    
    # Комбинируем условия
    df['atr_breakout_signal'] = np.logical_and.reduce(conditions).astype(int)
    
    # Добавляем отладочные колонки
    if DEBUG_MODE:
        for i, cond in enumerate(conditions):
            df[f'breakout_cond_{i}'] = cond.astype(int)
    
    return df


# %% [markdown]
# **Тренд Ишимоку**

# %%
def add_normalized_ichimoku(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_period: int = 52,
    normalization_base: str = "Close",
    replace_inf_with: float = None,
    fill_na: bool = True,
) -> pd.DataFrame:
    """
    Добавляет нормализованные компоненты Ichimoku без утечки будущего в исходный DataFrame.
    
    Параметры:
        df: DataFrame с колонками ['High', 'Low', 'Close'].
        tenkan_period: Период для Tenkan-sen (по умолчанию 9).
        kijun_period: Период для Kijun-sen (по умолчанию 26).
        senkou_period: Период для Senkou Span B (по умолчанию 52).
        normalization_base: Столбец для нормализации (по умолчанию 'Close').
        replace_inf_with: Если не None, заменяет inf на это значение (например, 0 или np.nan).
        fill_na: Заполнять ли пропуски последним валидным значением.
        
    Возвращает:
        Копию исходного DataFrame с добавленными нормализованными колонками Ichimoku.
    """
    # Проверка наличия необходимых колонок
    required_columns = {'High', 'Low', 'Close'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Не хватает колонок: {missing}")

    if normalization_base not in df.columns:
        raise ValueError(f"Столбец для нормализации '{normalization_base}' не найден.")

    # Создаем копию DataFrame, чтобы не изменять исходный
    df = df.copy()
    
    # Проверка на нули в normalization_base (чтобы не было деления на 0)
    if (df[normalization_base] == 0).any():
        if replace_inf_with is None:
            raise ValueError(
                f"В столбце '{normalization_base}' есть нули. "
                "Используйте `replace_inf_with` для обработки."
            )
        else:
            # Заменяем нули на очень маленькое число (1e-10), чтобы не было inf
            base = df[normalization_base].replace(0, 1e-10)
    else:
        base = df[normalization_base]

    # 1. Tenkan-sen (Conversion Line)
    tenkan_high = df['High'].rolling(tenkan_period, min_periods=1).max()
    tenkan_low = df['Low'].rolling(tenkan_period, min_periods=1).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    df['tenkan_sen_norm'] = tenkan_sen / base

    # 2. Kijun-sen (Base Line)
    kijun_high = df['High'].rolling(kijun_period, min_periods=1).max()
    kijun_low = df['Low'].rolling(kijun_period, min_periods=1).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    df['kijun_sen_norm'] = kijun_sen / base

    # 3. Senkou Span A (сдвиг назад, а не вперёд, чтобы избежать look-ahead)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(1) #kijun_period
    df['senkou_span_a_norm'] = senkou_span_a / base

    # 4. Senkou Span B (сдвиг назад)
    senkou_high = df['High'].rolling(senkou_period, min_periods=1).max()
    senkou_low = df['Low'].rolling(senkou_period, min_periods=1).min()
    senkou_span_b = ((senkou_high + senkou_low) / 2).shift(1) #kijun_period
    df['senkou_span_b_norm'] = senkou_span_b / base

    # 5. Границы облака
    #df['cloud_top_norm'] = df[['senkou_span_a_norm', 'senkou_span_b_norm']].max(axis=1)
    #df['cloud_bottom_norm'] = df[['senkou_span_a_norm', 'senkou_span_b_norm']].min(axis=1)

    # Замена inf (если replace_inf_with задан)
    if replace_inf_with is not None:
        df = df.replace([np.inf, -np.inf], replace_inf_with)

    # Заполнение пропусков (если fill_na=True)
    if fill_na:
        pd.set_option('future.no_silent_downcasting', True)  # <- Добавить эту строку
        df = df.ffill()  # Теперь без .infer_objects(), т.к. downcasting отключён

    return df


# %% [markdown]
# **Ишимоку ХТФ**

# %%
def add_normalized_ichimoku_htf(
    df: pd.DataFrame,
    tenkan_period: int = 108,
    kijun_period: int = 312,
    senkou_period: int = 624,
    normalization_base: str = "Close",
    replace_inf_with: float = 0,
    fill_na: bool = True,
) -> pd.DataFrame:
    """
    Добавляет нормализованные компоненты Ichimoku без утечки будущего в исходный DataFrame.
    
    Параметры:
        df: DataFrame с колонками ['High', 'Low', 'Close'].
        tenkan_period: Период для Tenkan-sen (по умолчанию 9).
        kijun_period: Период для Kijun-sen (по умолчанию 26).
        senkou_period: Период для Senkou Span B (по умолчанию 52).
        normalization_base: Столбец для нормализации (по умолчанию 'Close').
        replace_inf_with: Если не None, заменяет inf на это значение (например, 0 или np.nan).
        fill_na: Заполнять ли пропуски последним валидным значением.
        
    Возвращает:
        Копию исходного DataFrame с добавленными нормализованными колонками Ichimoku.
    """
    # Проверка наличия необходимых колонок
    required_columns = {'High', 'Low', 'Close'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Не хватает колонок: {missing}")

    if normalization_base not in df.columns:
        raise ValueError(f"Столбец для нормализации '{normalization_base}' не найден.")

    # Создаем копию DataFrame, чтобы не изменять исходный
    df = df.copy()
    
    # Проверка на нули в normalization_base (чтобы не было деления на 0)
    if (df[normalization_base] == 0).any():
        if replace_inf_with is None:
            raise ValueError(
                f"В столбце '{normalization_base}' есть нули. "
                "Используйте `replace_inf_with` для обработки."
            )
        else:
            # Заменяем нули на очень маленькое число (1e-10), чтобы не было inf
            base = df[normalization_base].replace(0, 1e-10)
    else:
        base = df[normalization_base]

    # 1. Tenkan-sen (Conversion Line)
    tenkan_high = df['High'].rolling(tenkan_period, min_periods=1).max()
    tenkan_low = df['Low'].rolling(tenkan_period, min_periods=1).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    df['tenkan_sen_norm_htf'] = tenkan_sen / base

    # 2. Kijun-sen (Base Line)
    kijun_high = df['High'].rolling(kijun_period, min_periods=1).max()
    kijun_low = df['Low'].rolling(kijun_period, min_periods=1).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    df['kijun_sen_norm_htf'] = kijun_sen / base

    # 3. Senkou Span A (сдвиг назад, а не вперёд, чтобы избежать look-ahead)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(1) #shift(kijun_period)
    df['senkou_span_a_norm_htf'] = senkou_span_a / base

    # 4. Senkou Span B (сдвиг назад)
    senkou_high = df['High'].rolling(senkou_period, min_periods=1).max()
    senkou_low = df['Low'].rolling(senkou_period, min_periods=1).min()
    senkou_span_b = ((senkou_high + senkou_low) / 2).shift(1) #shift(kijun_period)
    df['senkou_span_b_norm_htf'] = senkou_span_b / base

    # 5. Границы облака
    #df['cloud_top_norm_htf'] = df[['senkou_span_a_norm_htf', 'senkou_span_b_norm_htf']].max(axis=1)
    #df['cloud_bottom_norm_htf'] = df[['senkou_span_a_norm_htf', 'senkou_span_b_norm_htf']].min(axis=1)

    # Замена inf (если replace_inf_with задан)
    if replace_inf_with is not None:
        df = df.replace([np.inf, -np.inf], replace_inf_with)

    # Заполнение пропусков (если fill_na=True)
    if fill_na:
        pd.set_option('future.no_silent_downcasting', True)  # <- Добавить эту строку
        df = df.ffill()  # Теперь без .infer_objects(), т.к. downcasting отключён

    return df


# %% [markdown]
# **Ишимоку RSI**

# %%
def add_normalized_ichimoku_on_rsi(
    df: pd.DataFrame,
    rsi_period: int = 14,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_period: int = 52,
    replace_inf_with: float = None,
    fill_na: bool = True,
    prefix: str = "rsi14_ichimoku"
) -> pd.DataFrame:
    """
    Вычисляет Ichimoku на основе RSI и добавляет нормализованные компоненты в DataFrame.
    
    Параметры:
        df: DataFrame с колонкой 'Close'.
        rsi_period: Период RSI (по умолчанию 14).
        Остальные параметры — как в оригинальной Ichimoku-функции.
        prefix: Префикс для названий новых колонок.
    
    Возвращает:
        df с новыми колонками Ichimoku по RSI.
    """
    if 'Close' not in df.columns:
        raise ValueError("В DataFrame нет колонки 'Close' для расчета RSI.")

    df = df.copy()

    # Расчет RSI
    rsi = ta.rsi(df['Close'], length=rsi_period)
    df[f'RSI_{rsi_period}'] = rsi

    # Проверка на нули
    if (rsi == 0).any():
        if replace_inf_with is None:
            raise ValueError("RSI содержит нули, что может привести к делению на 0.")
        else:
            base = rsi.replace(0, 1e-10)
    else:
        base = rsi

    # Ichimoku на основе RSI
    # Tenkan
    tenkan_high = rsi.rolling(tenkan_period, min_periods=1).max()
    tenkan_low = rsi.rolling(tenkan_period, min_periods=1).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    df[f'{prefix}_tenkan_norm'] = tenkan_sen / base

    # Kijun
    kijun_high = rsi.rolling(kijun_period, min_periods=1).max()
    kijun_low = rsi.rolling(kijun_period, min_periods=1).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    df[f'{prefix}_kijun_norm'] = kijun_sen / base

    # Senkou A
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(1) #shift(kijun_period)
    df[f'{prefix}_senkou_a_norm'] = senkou_span_a / base

    # Senkou B
    senkou_high = rsi.rolling(senkou_period, min_periods=1).max()
    senkou_low = rsi.rolling(senkou_period, min_periods=1).min()
    senkou_span_b = ((senkou_high + senkou_low) / 2).shift(1) #shift(kijun_period)
    df[f'{prefix}_senkou_b_norm'] = senkou_span_b / base

    # Обработка inf
    if replace_inf_with is not None:
        df = df.replace([np.inf, -np.inf], replace_inf_with)

    # Пропуски
    if fill_na:
        pd.set_option('future.no_silent_downcasting', True)
        df = df.ffill()

    return df


# %% [markdown]
# **compute_atr_trailing**

# %%
def compute_atr_trailing(df, atr_period=14, multiplier=3):
    """
    Вычисляет ATR-based трейлинг-стоп и сигналы.
    Добавляет столбцы:
    - 'ATR' - Average True Range
    - 'TrailingStop' - трейлинг-стоп уровень
    - 'ATR_Signal' - бинарный сигнал (1/-1)
    - 'ATR_Signal_Norm' - нормализованный сигнал (0-1)
    - 'ATR_x10' - ATR * 10 (для наглядности)

    Параметры:
        df: DataFrame с колонками ['High', 'Low', 'Close']
        atr_period: период для ATR (по умолчанию 14)
        multiplier: множитель для ATR (по умолчанию 3)
    """
    df = df.copy()
    
    # Вычисляем True Range
    tr = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    
    # Вычисляем ATR с защитой от NaN/inf
    atr = tr.rolling(atr_period, min_periods=1).mean().replace([np.inf, -np.inf], np.nan).ffill()
    
    # Трейлинг-стоп
    df['TrailingStop'] = df['Close'] - multiplier * atr
    
    # Генерация сигнала
    df['ATR_Signal'] = np.where(df['Close'] > df['TrailingStop'].shift(1), 1, -1)
    
    # Нормализация сигнала от 0 до 1
    df['ATR_Signal_Norm'] = (df['ATR_Signal'] + 1) / 2  # преобразует -1/1 в 0/1
    
    # Дополнительный столбец ATR*10
    df['ATR_x10'] = atr * 10
    
    # Защита от оставшихся NaN (если они возникнут)
    df = df.replace([np.inf, -np.inf], np.nan).ffill()
    
    return df


# %% [markdown]
# **TRIX (Triple Exponential Average) и Elder Ray (Bull Power и Bear Power)**

# %%
def compute_trix_elder(df, trix_period=30, ema_period=50, eps=1e-8):
    df = df.copy()

    # === TRIX ===
    ema1 = df['Close'].ewm(span=trix_period, adjust=False).mean()
    ema2 = ema1.ewm(span=trix_period, adjust=False).mean()
    ema3 = ema2.ewm(span=trix_period, adjust=False).mean()
    df['TRIX'] = ema3.pct_change() * 100

    # === Elder Ray ===
    ema = df['Close'].ewm(span=ema_period, adjust=False).mean()
    df['BullPower'] = df['High'] - ema
    df['BearPower'] = df['Low'] - ema

    # === Обработка NaN/inf ===
    features = ['TRIX', 'BullPower', 'BearPower']
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    df[features] = df[features].ffill().bfill().fillna(0)

    # === Простая нормализация по модулю максимального значения ===
    for col in features:
        max_val = df[col].abs().max() + eps
        df[col] = df[col] / max_val

    return df


# %% [markdown]
# **Schaff Trend Cycle (STC) + Fisher Transform + Keltner Channel Width**

# %%
def compute_combined_indicators(df, 
                              stc_fast=23, 
                              stc_slow=50, 
                              stc_cycle=10,
                              fisher_length=10,
                              keltner_atr_period=20,
                              keltner_multiplier=2):
    """
    Улучшенная версия комбинированных индикаторов:
    - Schaff Trend Cycle (STC)
    - Fisher Transform
    - Keltner Channel Width
    
    Возвращает только новые признаки без колонок Date и target.
    Все значения нормализованы и защищены от NaN/Inf.
    """
    df = df.copy()
    
    # === 1. Улучшенный Schaff Trend Cycle ===
    def calculate_stc(close, fast, slow, cycle):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_ema = macd.ewm(span=cycle, adjust=False).mean()
        
        # Более стабильный расчет с защитой от деления на 0
        denominator = macd_ema.mask(macd_ema.abs() < 1e-10, np.nan)
        stc_raw = macd.sub(macd_ema).div(denominator)
        stc_raw = stc_raw.fillna(0).clip(-1, 1)
        
        return stc_raw.ewm(span=cycle, adjust=False).mean()
    
    df['STC'] = calculate_stc(df['Close'], stc_fast, stc_slow, stc_cycle)

    # === 2. Улучшенный Fisher Transform ===
    def calculate_fisher(high, low, length):
        hl2 = (high + low) / 2
        min_hl2 = hl2.rolling(length, min_periods=1).min()
        max_hl2 = hl2.rolling(length, min_periods=1).max()
        
        # Более безопасная нормализация
        range_hl2 = max_hl2 - min_hl2
        normalized = 2 * ((hl2 - min_hl2) / range_hl2.mask(range_hl2 < 1e-10, np.nan)) - 1
        normalized = normalized.fillna(0).clip(-0.999, 0.999)
        
        fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
        return fisher.ewm(span=length, adjust=False).mean()
    
    df['Fisher'] = calculate_fisher(df['High'], df['Low'], fisher_length)

    # === 3. Улучшенный Keltner Channel Width ===
    def calculate_keltner_width(high, low, close, atr_period, multiplier):
        typical_price = (high + low + close) / 3
        ema = typical_price.ewm(span=atr_period, adjust=False).mean()
        
        # Более точный расчет ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.ewm(span=atr_period, adjust=False).mean()
        return (2 * multiplier * atr) / ema  # Нормализованная ширина
    
    df['KeltnerWidth'] = calculate_keltner_width(
        df['High'], df['Low'], df['Close'], 
        keltner_atr_period, keltner_multiplier
    )

    # === Улучшенная обработка и нормализация ===
    features = ['STC', 'Fisher', 'KeltnerWidth']
    
    # Двухэтапная очистка
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    
    # Адаптивная нормализация (учет распределения)
    for col in features:
        q_low = df[col].quantile(0.01)
        q_high = df[col].quantile(0.99)
        df[col] = df[col].clip(q_low, q_high)
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-10)  # Z-score нормализация
    
    return df[features]


# %% [markdown]
# **RSI HTF**

# %%
def rsi_htf(df, timeframes=['1H'], rsi_periods=[7, 14], price_col='Close', date_col='Data'):
    """
    Добавляет в DataFrame нормализованные RSI с старших таймфреймов
    
    Параметры:
    df - исходный DataFrame
    timeframes - список таймфреймов для расчета (по умолчанию ['1H'])
    rsi_periods - списки периодов RSI (по умолчанию [7, 14])
    price_col - название столбца с ценами (по умолчанию 'Close')
    date_col - название столбца с датой (по умолчанию 'Data')
    """
    # Сохраняем исходное состояние DataFrame
    original_index = df.index
    original_columns = df.columns.tolist()
    was_index_datetime = isinstance(df.index, pd.DatetimeIndex)
    
    # Если 'Data' не в индексе или индекс не datetime, временно делаем datetime индексом
    if date_col in df.columns:
        df = df.set_index(date_col)
    
    # Убедимся, что индекс преобразован в DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    for tf in timeframes:
        # Ресемплинг на старший таймфрейм
        df_htf = df[[price_col]].resample(tf.lower()).last().dropna()
        
        # Расчет RSI для каждого периода
        for period in rsi_periods:
            col_name = f'RSI{period}_{tf}'
            df_htf[col_name] = ta.rsi(df_htf[price_col], length=period)
        
        # Объединение с исходным DataFrame
        df = df.join(df_htf[[f'RSI{period}_{tf}' for period in rsi_periods]], how='left')
        
        # Заполнение пропусков и нормализация
        for period in rsi_periods:
            col_name = f'RSI{period}_{tf}'
            norm_col = f'RSI{period}_{tf}_norm'
            
            df[col_name] = df[col_name].ffill()
            df[norm_col] = df[col_name] / 100
            df.drop(col_name, axis=1, inplace=True)
    
    # Восстанавливаем исходную структуру индекса и колонок
    if date_col not in original_index.names and date_col not in original_columns:
        df = df.reset_index()
    elif date_col in original_columns and date_col not in df.columns:
        df = df.reset_index()
    
    # Если исходный индекс не был datetime, возвращаем его как было
    if not was_index_datetime and date_col in df.columns:
        df = df.set_index(original_index.name if original_index.name else original_index)
    
    # Убедимся, что колонка 'Data' осталась на месте
    if date_col in original_columns and date_col not in df.columns:
        df[date_col] = pd.to_datetime(df.index if date_col == original_index.name else original_index)
    
    return df
# def rsi_htf(df, timeframes=['1H'], rsi_periods=[7, 14], price_col='Close', date_col='Data'):
#     """
#     Добавляет в DataFrame нормализованные RSI с старших таймфреймов
    
#     Параметры:
#     df - исходный DataFrame
#     timeframes - список таймфреймов для расчета (по умолчанию ['1H'])
#     rsi_periods - списки периодов RSI (по умолчанию [7, 14])
#     price_col - название столбца с ценами (по умолчанию 'Close')
#     date_col - название столбца с датой (по умолчанию 'Data')
#     """
#     # Сохраняем исходное состояние индекса
#     was_index = date_col in df.index.names
#     if not was_index:
#         df = df.set_index(date_col)
    
#     for tf in timeframes:
#         # Ресемплинг на старший таймфрейм (исправленная строка)
        
#         df_htf = df[[price_col]].resample(tf.lower()).last().dropna()
        
#         # Расчет RSI для каждого периода
#         for period in rsi_periods:
#             col_name = f'RSI{period}_{tf}'
#             df_htf[col_name] = ta.rsi(df_htf[price_col], length=period)
        
#         # Объединение с исходным DataFrame
#         df = df.join(df_htf[[f'RSI{period}_{tf}' for period in rsi_periods]], how='left')
        
#         # Заполнение пропусков и нормализация
#         for period in rsi_periods:
#             col_name = f'RSI{period}_{tf}'
#             norm_col = f'RSI{period}_{tf}_norm'
            
#             df[col_name] = df[col_name].ffill()
#             df[norm_col] = df[col_name] / 100
#             df.drop(col_name, axis=1, inplace=True)
    
#     # Восстанавливаем исходную структуру индекса
#     if not was_index:
#         df = df.reset_index()
    
#     return df


# %% [markdown]
# **Дивергенция RSI, Stoch, MACD, OBV и цены**

# %%
def calculate_smooth_divergence(df, 
                              window=14,
                              indicators={
                                  'RSI': {'window': 14},
                                  'Stoch': {'window': 14, 'k': 3, 'd': 3},
                                  'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
                                  'OBV': {}
                              }):
    """
    Возвращает столбец с плавными значениями дивергенции (-1 до 1).
    """
    df = df.copy()
    
    # 1. Расчет индикаторов
    # RSI
    if 'RSI' in indicators:
        df['RSI'] = ta.rsi(df['Close'], length=indicators['RSI']['window'])
    
    # Stochastic
    if 'Stoch' in indicators:
        stoch = ta.stoch(df['High'], df['Low'], df['Close'],
                        k=indicators['Stoch']['k'],
                        d=indicators['Stoch']['d'])
        stoch_col = [col for col in stoch.columns if 'STOCHk' in col][0]
        df['Stoch_K'] = stoch[stoch_col]
    
    # MACD
    if 'MACD' in indicators:
        macd = ta.macd(df['Close'],
                      fast=indicators['MACD']['fast'],
                      slow=indicators['MACD']['slow'],
                      signal=indicators['MACD']['signal'])
        macd_col = [col for col in macd.columns if 'MACD_' in col][0]
        df['MACD_line'] = macd[macd_col]
    
    # OBV
    if 'OBV' in indicators:
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        df['OBV_norm'] = (df['OBV'] - df['OBV'].rolling(window).mean()) / (df['OBV'].rolling(window).std() + 1e-10)
    
    # 2. Расчет дивергенции
    df['divergence'] = 0.0
    
    for i in range(window, len(df)):
        values = []
        weights = []
        
        # Собираем значения индикаторов
        if 'RSI' in df.columns:
            rsi_val = df['RSI'].iloc[i]
            values.append((rsi_val - 30) / (70 - 30))  # Нормализация RSI 30-70
            weights.append(0.35)
        
        if 'Stoch_K' in df.columns:
            stoch_val = df['Stoch_K'].iloc[i]
            values.append((stoch_val - 20) / (80 - 20))  # Нормализация Stochastic 20-80
            weights.append(0.25)
        
        if 'MACD_line' in df.columns:
            macd_val = df['MACD_line'].iloc[i]
            # Нормализация MACD относительно его экстремумов в окне
            macd_window = df['MACD_line'].iloc[i-window:i]
            if len(macd_window) > 0:
                macd_max = macd_window.max()
                macd_min = macd_window.min()
                if macd_max != macd_min:
                    macd_norm = (macd_val - macd_min) / (macd_max - macd_min)
                else:
                    macd_norm = 0
                values.append(macd_norm * 2 - 1)  # Приводим к диапазону [-1, 1]
                weights.append(0.25)
        
        if 'OBV_norm' in df.columns:
            obv_val = df['OBV_norm'].iloc[i]
            values.append(np.tanh(obv_val))  # Ограничиваем диапазон
            weights.append(0.15)
        
        # Рассчитываем дивергенцию только если есть минимум 2 индикатора
        if len(values) >= 2:
            # Взвешенное направление
            direction = np.sum(np.diff(values) * weights[:-1]) / np.sum(weights[:-1])
            
            # Мера дисперсии
            dispersion = (max(values) - min(values)) / (np.mean(np.abs(values)) + 1e-10)
            
            # Итоговое значение
            df.at[df.index[i], 'divergence'] = direction * dispersion
    
    # 3. Нормализация результатов
    max_div = df['divergence'].abs().max()
    if max_div > 0:
        df['divergence'] = df['divergence'] / max_div
    
    return df[['divergence']]


# %% [markdown]
# **EMA20, EMA50, VWMA20, HMA20, SMA100**

# %%
def different_MA(df, price_col='Close', eps=1e-8):
    price = df[price_col]
    volume = df['Volume']

    def normalize_diff(ma):
        return np.clip((ma - price) / (price + eps), -1, 1)

    # Скользящие средние
    ema50 = ta.ema(price, 50)
    ema200 = ta.ema(price, 200)
    sma100 = ta.sma(price, 100)
    sma200 = ta.sma(price, 200)

    # Углы наклона
    slope_ema50 = np.arctan(ema50.diff()) / np.pi
    slope_ema200 = np.arctan(ema200.diff()) / np.pi

    # Разности MA
    ema50_ema200_diff = ema50 - ema200
    sma100_sma200_diff = sma100 - sma200

    # Бинарные признаки пересечений
    ema50_above_ema200 = (ema50 > ema200).astype(int)
    ema50_cross_up = ((ema50 > ema200) & (ema50.shift(1) <= ema200.shift(1))).astype(int)
    ema50_cross_down = ((ema50 < ema200) & (ema50.shift(1) >= ema200.shift(1))).astype(int)

    return pd.DataFrame({
        # Базовые отклонения
        #'ema50_diff': normalize_diff(ema50),
        'ema200_diff': normalize_diff(ema200),
        #'sma100_diff': normalize_diff(sma100),
        #'sma200_diff': normalize_diff(sma200),
        
        # Взаимодействие MA
        'ema50_ema200_diff': ema50_ema200_diff,
        'sma100_sma200_diff': sma100_sma200_diff,
        'ema50_above_ema200': ema50_above_ema200,
        'ema50_cross_up': ema50_cross_up,
        'ema50_cross_down': ema50_cross_down,
        
        # Наклоны
        'ema50_slope': slope_ema50,
        'ema200_slope': slope_ema200,
        'ema50_slope_vs_ema200': slope_ema50 - slope_ema200
    })


# %% [markdown]
# # EMA

# %%
def different_EMA(df, price_col='Close', eps=1e-8):
    price = df[price_col]
    max_price = price.max()

    # EMA (по цене)
    ema5 = ta.ema(price, 5)
    ema20 = ta.ema(price, 20)
    ema50 = ta.ema(price, 50)
    ema100 = ta.ema(price, 100)
    ema200 = ta.ema(price, 200)

    # Нормализованные EMA
    ema20_norm = (ema20 - ema5)/ ema20
    ema50_norm = (ema50 - ema5)/ ema50
    ema100_norm = (ema100 - ema5)/ ema100
    ema200_norm = (ema200 - ema5)/ ema200

    # Углы наклона EMA (в радианах, нормализованные через π)
    slope_ema20 = np.arctan(ema20.diff(5)) / np.pi
    slope_ema50 = np.arctan(ema50.diff(5)) / np.pi
    slope_ema100 = np.arctan(ema100.diff(5)) / np.pi
    slope_ema200 = np.arctan(ema200.diff(5)) / np.pi

    # Разности между EMA (уже нормализованы)
    ema20_ema50_diff = ema20_norm - ema50_norm
    ema50_ema100_diff = ema50_norm - ema100_norm
    ema100_ema200_diff = ema100_norm - ema200_norm

    # Средняя цена за последние 5 свеч (среднее арифметическое Open, High, Low, Close)
    mean_price_5 = df[['Open', 'High', 'Low', 'Close']].mean(axis=1).rolling(5).mean()

    return pd.DataFrame({
        'ema20_norm': ema20_norm,
        'ema50_norm': ema50_norm,
        'ema100_norm': ema100_norm,
        'ema200_norm': ema200_norm,

        'ema20_slope': slope_ema20,
        'ema50_slope': slope_ema50,
        'ema100_slope': slope_ema100,
        'ema200_slope': slope_ema200,

        'ema20_ema50_diff': ema20_ema50_diff,
        'ema50_ema100_diff': ema50_ema100_diff,
        'ema100_ema200_diff': ema100_ema200_diff,
    })


# %% [markdown]
# ## EMA binary

# %%
def different_EMA_binary(df, price_col='Close'):
    df = df.copy()
    price = df[price_col]
    
    # EMA
    ema9 = ta.ema(price, 9)
    ema20 = ta.ema(price, 20)
    ema50 = ta.ema(price, 50)
    ema100 = ta.ema(price, 100)
    ema200 = ta.ema(price, 200)

    # Цена выше EMA
    df['price_above_ema9'] = (price > ema9).astype(int)
    df['price_above_ema20'] = (price > ema20).astype(int)
    df['price_above_ema50'] = (price > ema50).astype(int)
    df['price_above_ema100'] = (price > ema100).astype(int)
    df['price_above_ema200'] = (price > ema200).astype(int)

    # Пересечения EMA за последние 5 свечей
    def crossed(series1, series2, window=5):
        cross = ((series1 > series2) != (series1.shift(1) > series2.shift(1))).astype(int)
        return cross.rolling(window).max().fillna(0).astype(int)

    df['ema9_crossed_ema20_last5'] = crossed(ema9, ema20)
    df['ema20_crossed_ema50_last5'] = crossed(ema20, ema50)
    df['ema50_crossed_ema100_last5'] = crossed(ema50, ema100)
    df['ema100_crossed_ema200_last5'] = crossed(ema100, ema200)

    # Углы EMA и их рост (в радианах / pi)
    def slope_increased(series, window=5):
        slope = np.arctan(series.diff(window)) / np.pi
        return (slope > slope.shift(window)).astype(int)

    df['ema20_slope_increased'] = slope_increased(ema20)
    df['ema50_slope_increased'] = slope_increased(ema50)
    df['ema100_slope_increased'] = slope_increased(ema100)
    df['ema200_slope_increased'] = slope_increased(ema200)

    # ema20 и ema50 оба растут и расходятся
    slope_20 = ema20.diff(5)
    slope_50 = ema50.diff(5)
    spread = ema20 - ema50
    spread_change = spread.diff(5)
    df['ema20_and_50_rising_diverging'] = ((slope_20 > 0) & (slope_50 > 0) & (spread_change > 0)).astype(int)

    return df


# %% [markdown]
# # Объемные индикаторы

# %%
def volume_base_indicators(df, length=28, short_window=5):
    """
    Добавляет в DataFrame нормализованные индикаторы объема:
    - OBV: нормализованный через Z-скор (динамика изменения)
    - Volume MA: отношение к текущему объёму (не к максимуму!)
    - A/D_cum: процентное изменение за short_window свечей
    - Force Index: стандартизированный (Z-норма)
    
    Параметры:
        df: DataFrame с колонками ['Close', 'High', 'Low', 'Volume']
        length: период для скользящих окон
        short_window: окно для краткосрочных изменений (по умолчанию 5)
    """
    # 1. On-Balance Volume (OBV) с Z-нормализацией
    obv = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    df['obv_zscore'] = (obv - obv.rolling(length).mean()) / obv.rolling(length).std()
    
    # 2. Volume MA: отношение текущего объёма к его скользящей средней
    df['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(length).mean()
    
    # 3. A/D Cum: процентное изменение за short_window свечей
    ad = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'].replace(0, 0.0001)) * df['Volume']
    ad_cum = ad.rolling(length).sum()
    df['ad_cum_pct_change'] = ad_cum.pct_change(short_window)  # % изменение за N свечей
    
    # 4. Force Index: стандартизированный
    force_index = df['Close'].diff() * df['Volume']
    df['force_index_z'] = (force_index - force_index.rolling(length).mean()) / force_index.rolling(length).std()
    
    return df.dropna()


# %% [markdown]
# ## volume_base_indicators_binary

# %%
def volume_base_indicators_binary(df: pd.DataFrame,
                                   obv_ema_period: int = 20,
                                   ad_ema_period: int = 20,
                                   force_ema_period: int = 13,
                                   vol_ma_period: int = 20,
                                   vol_ma_type: str = 'ema') -> pd.DataFrame:
    """
    Добавляет бинарные признаки объёмных индикаторов:
    - OBV выше своей EMA
    - A/D выше своей EMA
    - Force Index выше своей EMA
    - Текущий объём выше MA объёма
    
    Параметры:
    - *_period: период сглаживания
    - vol_ma_type: 'ema' или 'sma' для volume_ma_ratio
    """
    df = df.copy()
    eps = 1e-8
    
    # Проверка нужных колонок
    required_cols = ['Close', 'High', 'Low', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"В DataFrame отсутствует обязательная колонка: {col}")

    # --- OBV ---
    obv = np.where(df['Close'] > df['Close'].shift(), df['Volume'],
          np.where(df['Close'] < df['Close'].shift(), -df['Volume'], 0))
    obv = pd.Series(obv).cumsum()
    obv_ema = obv.ewm(span=obv_ema_period, adjust=False).mean()
    df[f'obv_gt_ema{obv_ema_period}'] = (obv > obv_ema).astype(int)

    # --- A/D Line ---
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / ((df['High'] - df['Low']).replace(0, eps))
    mfv = mfm * df['Volume']
    ad = mfv.cumsum()
    ad_ema = ad.ewm(span=ad_ema_period, adjust=False).mean()
    df[f'ad_gt_ema{ad_ema_period}'] = (ad > ad_ema).astype(int)

    # --- Force Index ---
    force_index = df['Close'].diff() * df['Volume']
    force_ema = force_index.ewm(span=force_ema_period, adjust=False).mean()
    df[f'force_gt_ema{force_ema_period}'] = (force_index > force_ema).astype(int)

    # --- Volume MA ratio ---
    if vol_ma_type.lower() == 'ema':
        vol_ma = df['Volume'].ewm(span=vol_ma_period, adjust=False).mean()
        suffix = f'ema{vol_ma_period}'
    else:
        vol_ma = df['Volume'].rolling(window=vol_ma_period).mean()
        suffix = f'sma{vol_ma_period}'

    df[f'volume_gt_{suffix}'] = (df['Volume'] > vol_ma).astype(int)

    return df

# df = volume_base_indicators_binary(df,obv_ema_period=21,ad_ema_period=14,force_ema_period=13,vol_ma_period=20,vol_ma_type='sma')


# %% [markdown]
# ## bullish_volume_dominance_binary

# %%
def bullish_volume_dominance_binary(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = df.copy()

    if not all(col in df.columns for col in ['Open', 'Close', 'Volume']):
        raise ValueError("DataFrame должен содержать колонки 'Open', 'Close', 'Volume'")

    bullish = (df['Close'] > df['Open']).astype(int)
    bearish = (df['Close'] < df['Open']).astype(int)

    bullish_volume = df['Volume'] * bullish
    bearish_volume = df['Volume'] * bearish

    bullish_sum = bullish_volume.rolling(window).sum()
    bearish_sum = bearish_volume.rolling(window).sum()

    column_name = f'bullish_volume_dominance_{window}'
    df[column_name] = (bullish_sum > bearish_sum).astype(int)

    return df  # теперь вернётся df со всеми колонками


# df = bullish_volume_dominance_binary(df, window=10)

# %% [markdown]
# # vwap

# %%
def VWAP(df: pd.DataFrame,
         slope_period: int = 28,
         slope_column_name: str = 'VWAP_slope') -> pd.DataFrame:
    """
    Добавляет VWAP-индикаторы в DataFrame:
    - vwap_intraday: VWAP со сбросом на полночь UTC
    - vwap_ema6h: EMA от VWAP по 6 часам
    - vwap_intraday_norm: нормализация через разницу с vwap_5
    - vwap_ema6h_norm: нормализация через разницу с vwap_5
    - VWAP_slope: процентное изменение VWAP за N периодов
    """

    df = df.copy()
    eps = 1e-8

    if not all(col in df.columns for col in ['Data', 'Close', 'Volume']):
        raise ValueError("DataFrame должен содержать колонки 'Data', 'Close', 'Volume'")
    
    df['datetime'] = pd.to_datetime(df['Data']).dt.tz_localize('UTC')
    df['date_only'] = df['datetime'].dt.date

    # VWAP
    df['tp'] = df['Close']
    df['tpv'] = df['tp'] * df['Volume']
    df['cum_tpv'] = df.groupby('date_only')['tpv'].cumsum()
    df['cum_vol'] = df.groupby('date_only')['Volume'].cumsum().replace(0, eps)
    df['vwap_intraday'] = df['cum_tpv'] / df['cum_vol']

    # EMA от VWAP (6ч = 360 свеч)
    df['vwap_ema6h'] = df['vwap_intraday'].ewm(span=360, adjust=False).mean()

    # Быстрый VWAP для нормализации
    df['vwap_5'] = df['vwap_intraday'].ewm(span=5, adjust=False).mean()

    # Альтернативная нормализация (как ты делал для EMA/ATR/Keltner)
    df['vwap_intraday_norm'] = (df['vwap_intraday'] - df['vwap_5']) / (df['vwap_intraday'] + eps)
    df['vwap_ema6h_norm'] = (df['vwap_ema6h'] - df['vwap_5']) / (df['vwap_ema6h'] + eps)

    # Наклон VWAP в процентах
    df[slope_column_name] = df['vwap_intraday'].pct_change(periods=slope_period) * 100
    df[slope_column_name] = df[slope_column_name].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Удаление временных колонок
    df.drop(columns=[
        'datetime', 'date_only', 'tp', 'tpv', 'cum_tpv', 'cum_vol',
        'vwap_intraday', 'vwap_ema6h', 'vwap_5'
    ], inplace=True)

    return df


# %% [markdown]
# ## vwap binary

# %%
def VWAP_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет два бинарных признака:
    - price_above_vwap_intraday: цена выше VWAP
    - price_above_vwap_ema6h: цена выше EMA от VWAP (6 часов)
    """
    df = df.copy()
    eps = 1e-8

    if not all(col in df.columns for col in ['Data', 'Close', 'Volume']):
        raise ValueError("DataFrame должен содержать колонки 'Data', 'Close', 'Volume'")
    
    df['datetime'] = pd.to_datetime(df['Data']).dt.tz_localize('UTC')
    df['date_only'] = df['datetime'].dt.date

    # VWAP (по Close)
    df['tpv'] = df['Close'] * df['Volume']
    df['cum_tpv'] = df.groupby('date_only')['tpv'].cumsum()
    df['cum_vol'] = df.groupby('date_only')['Volume'].cumsum().replace(0, eps)
    vwap_intraday = df['cum_tpv'] / df['cum_vol']

    # EMA от VWAP (6 часов = 360 свечей при 1м таймфрейме)
    vwap_ema6h = vwap_intraday.ewm(span=360, adjust=False).mean()

    # Бинарные признаки
    df['price_above_vwap_intraday'] = (df['Close'] > vwap_intraday).astype(int)
    df['price_above_vwap_ema6h'] = (df['Close'] > vwap_ema6h).astype(int)

    # Удалим временные и служебные колонки
    df.drop(columns=['datetime', 'date_only', 'tpv', 'cum_tpv', 'cum_vol'], inplace=True)

    return df


# %% [markdown]
# ## VWAP updates: add_vwap_features

# %%
def add_vwap_features_with_norm(df: pd.DataFrame,
                                 ema_span: int = 9,
                                 vwap_ema_span: int = 360,
                                 zscore_window: int = 100,
                                 rsi_period: int = 14,
                                 debug: bool = False) -> pd.DataFrame:
    """
    Добавляет три признака:
    - vwap_ema6h_zscore: z-score от vwap_ema6h_norm
    - vwap_price_distance: отклонение ema9 от vwap_ema6h_norm
    - RSI_vwap_divergence: расхождение RSI и направления vwap_ema6h_norm
    """

    df = df.copy()
    eps = 1e-8

    try:
        if not all(col in df.columns for col in ['Close', 'Volume', 'Data']):
            raise ValueError("В df должны быть колонки: 'Close', 'Volume', 'Data' (временные метки)")

        # EMA9 от цены
        df['ema9'] = df['Close'].ewm(span=ema_span, adjust=False).mean()

        # VWAP: интрадеевый по UTC
        df['datetime'] = pd.to_datetime(df['Data']).dt.tz_localize('UTC')
        df['date_only'] = df['datetime'].dt.date

        df['tpv'] = df['ema9'] * df['Volume']
        df['cum_tpv'] = df.groupby('date_only')['tpv'].cumsum()
        df['cum_vol'] = df.groupby('date_only')['Volume'].cumsum().replace(0, eps)
        df['vwap_intraday'] = df['cum_tpv'] / df['cum_vol']

        # EMA от VWAP
        df['vwap_ema6h'] = df['vwap_intraday'].ewm(span=vwap_ema_span, adjust=False).mean()
        df['vwap_5'] = df['vwap_intraday'].ewm(span=5, adjust=False).mean()

        # Нормализация
        df['vwap_ema6h_norm'] = (df['vwap_ema6h'] - df['vwap_5']) / (df['vwap_ema6h'] + eps)

        # Признак 1: z-score нормализованного VWAP
        mean_norm = df['vwap_ema6h_norm'].rolling(zscore_window).mean()
        std_norm = df['vwap_ema6h_norm'].rolling(zscore_window).std().replace(0, eps)
        df['vwap_ema6h_zscore'] = (df['vwap_ema6h_norm'] - mean_norm) / std_norm

        # Признак 2: отклонение ema9 от нормализованного VWAP
        df['vwap_price_distance_direct'] = np.sign(df['ema9'] - df['vwap_ema6h']) * (np.abs(df['ema9'] - df['vwap_ema6h']) / df['Close'].rolling(100).std().clip(lower=eps))
        df['vwap_price_distance'] = ((df['ema9'] - df['vwap_ema6h']) - (df['ema9'] - df['vwap_ema6h']).rolling(100).mean()) / ((df['ema9'] - df['vwap_ema6h']).rolling(100).std() + eps)

        # RSI по ema9
        delta = df['ema9'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(rsi_period).mean()
        avg_loss = loss.rolling(rsi_period).mean()
        rs = avg_gain / (avg_loss + eps)
        df['rsi'] = 100 - (100 / (1 + rs))

        # Признак 3: RSI vs VWAP
        df['RSI_vwap_divergence'] = ((df['rsi'] - 50) / 50) - np.sign(df['vwap_ema6h_norm'])

        # Убираем временные и промежуточные колонки
        df.drop(columns=[
            'datetime', 'date_only', 'tpv', 'cum_tpv', 'cum_vol',
            'vwap_intraday', 'vwap_ema6h', 'vwap_5', 'rsi','ema9'
        ], inplace=True)

        return df

    except Exception as e:
        if debug:
            print(f"❌ Ошибка в add_vwap_features_with_norm: {e}")
        return df


# %% [markdown]
# **liquidity_imbalance**

# %%
def add_liquidity_imbalance(df, period=20, inplace=False):
    if not inplace:
        df = df.copy()
    
    # Разница между покупками и продажами (упрощенная версия)
    df['liq_imb'] = (2*df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low'] + 1e-10) * np.log1p(df['Volume'])
    
    # Скользящая нормализация
    rolling_max = df['liq_imb'].abs().rolling(period).max()
    df['liq_imb'] = df['liq_imb'] / (rolling_max + 1e-10)
    
    return df


# %% [markdown]
# **hidden_divergence**

# %%
def add_hidden_divergence(df, rsi_period=14, inplace=False):
    if not inplace:
        df = df.copy()
    
    # Классический RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Скрытая дивергенция
    price_higher = df['Close'] > df['Close'].shift(1)
    rsi_lower = rsi < rsi.shift(1)
    df['hidden_div'] = (
    (price_higher & rsi_lower).astype(int) - 
    ((df['Close'] < df['Close'].shift(1)) & (rsi > rsi.shift(1))).astype(int)
)
    
    return df


# %% [markdown]
# **Ускорение цены**

# %%
def add_price_acceleration(df, window=5):

    # Создаем копию DataFrame чтобы не менять исходные данные
    result_df = df.copy()
    
    # Рассчитываем ускорение
    velocity = result_df['Close'].diff(1)
    acceleration = velocity.diff(1)
    smoothed_accel = acceleration.rolling(window=window).mean()
    
    # Нормализация через tanh
    price_std = result_df['Close'].pct_change().std()
    if price_std > 0:
        normalized_accel = np.tanh(smoothed_accel / price_std)
    else:
        normalized_accel = smoothed_accel * 0  # Если волатильность нулевая
    
    # Добавляем столбец
    result_df['Acceleration'] = normalized_accel
    
    return result_df

# Пример использования
# new_df = add_price_acceleration(df)


# %%

# %% [markdown]
# **Модели сессий**

# %% [markdown]
# Сессии

# %%
def add_trading_sessions(df):
    df_sessions = df.copy()
    df_sessions['Date'] = pd.to_datetime(df_sessions['Date'])
    
    # Векторизованное определение DST для всех дат сразу
    tz_london = timezone('Europe/London')
    tz_ny = timezone('America/New_York')
    
    # Локализуем даты в соответствующих временных зонах
    london_dates = df_sessions['Date'].dt.tz_localize(tz_london, ambiguous='NaT', nonexistent='NaT')
    ny_dates = df_sessions['Date'].dt.tz_localize(tz_ny, ambiguous='NaT', nonexistent='NaT')
    
    # Преобразуем в серии datetime и применяем dst()
    df_sessions['London_DST'] = london_dates.apply(lambda x: x.dst().total_seconds() if pd.notna(x) else 0) != 0
    df_sessions['NewYork_DST'] = ny_dates.apply(lambda x: x.dst().total_seconds() if pd.notna(x) else 0) != 0
    
    hour = df_sessions['Date'].dt.hour
    minute = df_sessions['Date'].dt.minute
    
    # Asia (03:00–10:00 МСК зимой, 02:00–09:00 летом)
    df_sessions['Asia'] = (
        (~df_sessions['London_DST'] & (hour >= 3) & (hour < 10)) | 
        (df_sessions['London_DST'] & (hour >= 2) & (hour < 9))
    ).astype(int)
    
    # Frankfurt (10:00–11:00 МСК зимой, 09:00–10:00 летом)
    df_sessions['Frankfurt'] = (
        (~df_sessions['London_DST'] & (hour >= 10) & (hour < 11)) | 
        (df_sessions['London_DST'] & (hour >= 9) & (hour < 10))
    ).astype(int)
    
    # London (11:00–20:00 МСК зимой, 10:00–19:00 летом)
    df_sessions['London'] = (
        (~df_sessions['London_DST'] & (hour >= 11) & (hour < 13)) | 
        (df_sessions['London_DST'] & (hour >= 10) & (hour < 12))
    ).astype(int)
    
    # NewYork (16:00–01:00 МСК зимой, 15:00–00:00 летом)
    df_sessions['NewYork'] = (
        (~df_sessions['NewYork_DST'] & ((hour >= 16) | (hour < 1))) | 
        (df_sessions['NewYork_DST'] & ((hour >= 15) | (hour < 0)))
    ).astype(int)
    
    # Lunch (07:00–08:00 МСК зимой, 06:00–07:00 летом)
    df_sessions['Lunch'] = (
        (~df_sessions['London_DST'] & ((hour == 13) | ((hour == 16) & (minute == 0)))) | 
        (df_sessions['London_DST'] & ((hour == 12) | ((hour == 15) & (minute == 0))))
    ).astype(int)
    
    df_sessions.drop(['London_DST', 'NewYork_DST'], axis=1, inplace=True)
    
    return df_sessions

# Пример использования:
# df = add_trading_sessions(df)


# %% [markdown]
# Снятие сессий и дистанция

# %%
def session_distance(df):
    sessions = ['Asia', 'Frankfurt', 'London', 'NewYork', 'Lunch']
    df = df.copy()
    
    for session in sessions:
        # Находим моменты начала сессий
        session_starts = (df[session] == 1) & (df[session].shift(1) != 1)
        
        # Создаем столбцы с экстремумами предыдущей сессии
        prev_high = df['High'].where(session_starts).ffill()
        prev_low = df['Low'].where(session_starts).ffill()
        
        # Вычисляем диапазон предыдущей сессии (избегаем деления на 0)
        prev_range = np.where((prev_high - prev_low) != 0, 
                             prev_high - prev_low, 
                             np.nan)
        
        # Вычисляем расстояния
        distance = np.zeros(len(df))
        
        # Пробитие сверху (используем High текущей свечи)
        above_mask = (df['High'] > prev_high) & (df[session] == 1)
        distance[above_mask] = (df['High'] - prev_high)[above_mask] / prev_range[above_mask]
        
        # Пробитие снизу (используем Low текущей свечи)
        below_mask = (df['Low'] < prev_low) & (df[session] == 1)
        distance[below_mask] = (df['Low'] - prev_low)[below_mask] / prev_range[below_mask]
        
        # Ограничиваем значения и добавляем в DataFrame
        df[f'{session}_Distance'] = np.clip(distance, -1, 1)
    
    return df


# %% [markdown]
# **Снятие фракталов**

# %%
def sfp_fractals(df):
    df = df.copy()
    df['fractals_broken_count'] = 0
    df['fractals_broken_ratio'] = -1.0
    
    # Предварительно вычисляем все фракталы (3-свечные минимумы) один раз
    lows = df['Low'].values
    fractal_lows = []
    for i in range(1, len(df)-1):
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            fractal_lows.append((i, lows[i]))
    
    # Преобразуем в numpy массив для быстрой обработки
    fractal_indices = np.array([x[0] for x in fractal_lows])
    fractal_values = np.array([x[1] for x in fractal_lows])
    
    # Основной цикл
    for i in range(len(df)):
        start_idx = max(0, i - 14)
        
        # Быстрая фильтрация фракталов в окне с использованием numpy
        mask = (fractal_indices >= start_idx) & (fractal_indices < i)
        window_indices = fractal_indices[mask]
        window_values = fractal_values[mask]
        
        if len(window_indices) == 0:
            continue
            
        broken_count = 0
        ratios = []
        
        # Проверяем последние 3 свечи + текущую
        check_lows = lows[max(0, i-3):i+1]
        
        for j in range(len(window_values)):
            if np.any(check_lows < window_values[j]):
                broken_count += 1
                
                # Рассчитываем ratio только для самого нижнего фрактала
                if window_values[j] == np.min(window_values):
                    fractal_idx = window_indices[j]
                    min_after = np.min(lows[fractal_idx+1:i+1])
                    max_after = np.max(df['High'].values[fractal_idx+1:i+1])
                    current_close = df['Close'].iloc[i]
                    
                    if current_close > window_values[j]:
                        ratio = (current_close - window_values[j]) / (max_after - window_values[j]) if max_after > window_values[j] else 0.0
                    else:
                        ratio = (current_close - window_values[j]) / (window_values[j] - min_after) if min_after < window_values[j] else 0.0
                    ratios.append(ratio)
        
        if broken_count > 0:
            df.at[i, 'fractals_broken_count'] = broken_count
            df.at[i, 'fractals_broken_ratio'] = np.clip(np.mean(ratios), -1, 1) if ratios else 0.0
    
    return df


# %% [markdown]
# **Kagi Phase Cloud**

# %%
def kagi_conversion_line(df, conversion_period=9, baseline_period=26, vortex_period=14):
    """
    Добавляет в DataFrame столбцы:
    1. 'kagi_conversion' (норм. -1 до 1) - аналог Tenkan-sen на основе Kagi
    2. 'phase_baseline' (норм. 0 до 1) - аналог Kijun-sen с Theta-фильтром
    3. 'vortex_cloud' (норм. -1 до 1) - разница VI+/VI- (направление облака)
    4. 'kagi_phase_diff' (норм. -1 до 1) - разница между 1 и 2 (трендовый импульс)
    """
    df = df.copy()
    
    # 1. Kagi Conversion Line (аналог Tenkan)
    def _kagi_line(close, threshold_series):
        kagi = [close.iloc[0]]
        direction = 1  # 1 = up, -1 = down
        
        for i in range(1, len(close)):
            current_threshold = threshold_series.iloc[i]  # Берем текущее значение ATR
            move = close.iloc[i] - kagi[-1]
            
            if (direction == 1 and move > 0) or (direction == -1 and move < 0):
                kagi.append(kagi[-1] + move)
            elif abs(move) >= current_threshold:
                direction *= -1
                kagi.append(kagi[-1] + move)
            else:
                kagi.append(kagi[-1])
                
        return pd.Series(kagi, index=close.index)
    
    # Вычисляем ATR для Kagi
    atr = (df['High'] - df['Low']).rolling(conversion_period).mean()
    kagi = _kagi_line(df['Close'], threshold_series=1.5*atr)
    df['kagi_conversion'] = savgol_filter(kagi, window_length=conversion_period, polyorder=2)
    
    # 2. Phase Baseline (аналог Kijun с Theta-фильтром)
    def _theta_filter(series, period):
        theta = [series.iloc[:period].mean()]
        for i in range(period, len(series)):
            drift = (series.iloc[i-period:i].mean() - theta[-1]) / period
            theta.append(theta[-1] + drift + 0.5*(series.iloc[i] - series.iloc[i-period]))
        return pd.Series(theta, index=series.index[period-1:])
    
    median_price = (df['High'] + df['Low']) / 2
    theta_baseline = _theta_filter(median_price, baseline_period)
    df['phase_baseline'] = theta_baseline.reindex(df.index, method='ffill')
    
    # 3. Vortex Cloud (на основе VI+ и VI-)
    tr = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    vm_plus = abs(df['High'] - df['Low'].shift(1))
    vm_minus = abs(df['Low'] - df['High'].shift(1))
    
    vi_plus = vm_plus.rolling(vortex_period).sum() / tr.rolling(vortex_period).sum()
    vi_minus = vm_minus.rolling(vortex_period).sum() / tr.rolling(vortex_period).sum()
    df['vortex_cloud'] = vi_plus - vi_minus  # разница для направления
    
    # 4. Комбинированный столбец (разница Kagi и Baseline)
    df['kagi_phase_diff'] = df['kagi_conversion'] - df['phase_baseline']
    
    # Нормализация (-1 до 1 для трендовых, 0-1 для уровневых)
    for col in ['kagi_conversion', 'vortex_cloud', 'kagi_phase_diff']:
        df[col] = 2 * (df[col] - df[col].rolling(100).min()) / (
            df[col].rolling(100).max() - df[col].rolling(100).min() + 1e-10) - 1
    
    df['phase_baseline'] = (df['phase_baseline'] - df['phase_baseline'].rolling(100).min()) / (
        df['phase_baseline'].rolling(100).max() - df['phase_baseline'].rolling(100).min() + 1e-10)
    
    # Заполнение NaN (первые 100 баров)
    df.fillna(method='bfill', inplace=True)
    
    return df


# %% [markdown]
# **Слом структуры**

# %%
def add_structure_break_long(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Быстрая версия: ищет сломы структуры в long без тяжелых циклов.
    """
    df = df.copy()
    df['structure_break_long'] = 0

    # 1. Найти high-фракталы Вильямса
    is_fractal = (
        (df['High'] > df['High'].shift(1)) &
        (df['High'] > df['High'].shift(2)) &
        (df['High'] > df['High'].shift(-1)) &
        (df['High'] > df['High'].shift(-2))
    )
    fractal_indices = np.where(is_fractal)[0]
    fractal_highs = df['High'].values

    # 2. Создаем бинарный вектор длиной df, где 1 — если был слом в этом баре
    break_marks = np.zeros(len(df), dtype=int)

    for idx in fractal_indices:
        # Проверка выхода за границы
        if idx + 1 >= len(df):
            continue

        # Закрытия после фрактала
        post_close = df['Close'].values[idx+1:]
        break_found = np.where(post_close > fractal_highs[idx])[0]

        if break_found.size > 0:
            break_idx = idx + 1 + break_found[0]
            break_marks[break_idx] = 1

    # 3. Для каждого бара проверим, были ли "сломы" в предыдущих window барах
    rolling_breaks = pd.Series(break_marks).rolling(window=window, min_periods=1).max().fillna(0).astype(int)
    df['structure_break_long'] = rolling_breaks.values

    return df


# %% [markdown]
# **Liquidity Imbalance Short-Squeeze Score" (LISS)**

# %%
def safe_fisher_transform(series, epsilon=1e-7):
    """Устойчивое преобразование Фишера без inf значений"""
    series = np.clip(series, -1 + epsilon, 1 - epsilon)  # ограничиваем (-1, 1)
    return 0.5 * np.log((1 + series) / (1 - series))

def calculate_LISS(df):
    # Нормализуем объем (если ещё не нормализован)
    volume_norm = df['Volume'] / df['Volume'].rolling(20).max().replace(0, 1)
    
    # Устойчивый LISS
    LISS = (2 * df['VP_Norm']) / (1 + expit(-df['atr_14_norm'])) - (df['STC'] * safe_fisher_transform(volume_norm))
    
    # Масштабируем в [0, 1] для удобства
    LISS = (LISS - LISS.min()) / (LISS.max() - LISS.min() + 1e-10)
    return LISS


# %% [markdown]
# Тест группы новых индикаторов

# %%
eps = 1e-8

# 1. Tenkan-sen Breakout (Ишимоку)
def add_tenkan_breakout(df, period=9):
    df['tenkan_sen'] = (df['High'].rolling(period).max().shift(1) + 
                        df['Low'].rolling(period).min().shift(1)) / 2
    df['tenkan_breakout'] = ((df['Close'] > df['tenkan_sen']) & 
                            (df['Close'].shift(1) <= df['tenkan_sen'].shift(1))).astype(int)
    return df

# 2. RSI Divergence (без look-ahead)
def add_rsi_divergence(df, window=14, lookback=5):
    eps = 1e-8
    
    # Заменяем RSIIndicator на pandas_ta версию
    rsi_values = rsi(close=df['Close'], length=window)
    
    df['rsi_low'] = rsi_values.rolling(lookback).min().shift(1)
    df['price_low'] = df['Low'].rolling(lookback).min().shift(1)
    df['bullish_div'] = ((df['rsi_low'].diff() > 0) & 
                        (df['price_low'].diff() < 0)).astype(int)
    
    return df

# 3. Fisher Transform (без look-ahead)
def add_fisher_transform(df, period=10):
    hl2 = (df['High'] + df['Low']) / 2
    max_hl2 = hl2.rolling(period).max().shift(1)  # shift(1) исключает текущую свечу
    min_hl2 = hl2.rolling(period).min().shift(1)
    normalized = (hl2 - min_hl2) / (max_hl2 - min_hl2 + eps)
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized + eps))
    df['fisher'] = fisher.clip(-1, 1)
    return df

# 4. False Breakout Detector (уже в целом правильно, но можно уточнить)
def add_false_breakout(df, window=10):
    df['high_prev'] = df['High'].rolling(window).max().shift(1)
    df['low_prev'] = df['Low'].rolling(window).min().shift(1)
    df['fake_bull'] = ((df['Low'] < df['low_prev']) & 
                      (df['Close'] > df['low_prev'])).astype(int)
    return df

# 5. H1 Trend (без look-ahead)
def add_h1_trend(df, period=12):
    sma_h1 = df['Close'].rolling(period).mean().shift(1)  # shift(1) исключает текущую свечу
    trend = (df['Close'] - sma_h1) / (sma_h1 + eps)
    df['h1_trend'] = trend.clip(-1, 1)
    return df

# 6. ATR Ratio (без look-ahead)
def add_atr_ratio(df, short_period=14, long_period=12):
    eps = 1e-8

    atr_values = atr(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        length=short_period
    )
    
    atr_ratio = atr_values / (atr_values.rolling(long_period).mean().shift(1) + eps)
    df['atr_ratio'] = atr_ratio.clip(0, 2) / 2
    
    return df

# 7. Long-term Liquidity Level (без look-ahead)
def add_liquidity_distance(df, window=1000):
    rounded = df['Close'].round(2)  # округление повышает шанс повторов
    liq = rounded.rolling(window).apply(
        lambda x: x.value_counts().idxmax() if not x.value_counts().empty else np.nan
    ).shift(1)
    df['liq_distance'] = ((df['Close'] - liq) / (liq + eps)).clip(-1, 1)
    return df

# ---------------------------------------------------
# Применяем все функции к DataFrame
# ---------------------------------------------------
def add_all_features(df):
    start_total = time.time()
    
    # 1. Tenkan Breakout
    start = time.time()
    df = add_tenkan_breakout(df)
    #print(f"Tenkan Breakout выполнена за {time.time() - start:.2f} сек")
    
    # 2. RSI Divergence
    start = time.time()
    df = add_rsi_divergence(df)
    #print(f"RSI Divergence выполнена за {time.time() - start:.2f} сек")
    
    # 3. Fisher Transform
    start = time.time()
    df = add_fisher_transform(df)
    #print(f"Fisher Transform выполнена за {time.time() - start:.2f} сек")
    
    # 4. False Breakout
    start = time.time()
    df = add_false_breakout(df)
    #print(f"False Breakout выполнена за {time.time() - start:.2f} сек")
    
    # 5. H1 Trend
    start = time.time()
    df = add_h1_trend(df)
    #print(f"H1 Trend выполнена за {time.time() - start:.2f} сек")
    
    # 6. ATR Ratio
    start = time.time()
    df = add_atr_ratio(df)
    #print(f"ATR Ratio выполнена за {time.time() - start:.2f} сек")
    
    # # 7. Liquidity Distance
    # start = time.time()
    # df = add_liquidity_distance(df)
    # #print(f"Liquidity Distance выполнена за {time.time() - start:.2f} сек")
    
    # Удаляем промежуточные колонны
    start = time.time()
    cols_to_drop = ['tenkan_sen', 'rsi_low', 'price_low', 'high_prev', 'low_prev', 'liq_level']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    #print(f"Очистка промежуточных данных выполнена за {time.time() - start:.2f} сек")
    
    #print(f"\nВсе операции выполнены за {time.time() - start_total:.2f} сек")
    
    return df

# Пример использования:
# df = pd.read_csv('your_data.csv')
# df = add_all_features(df)


# %% [markdown]
# **Теория четвертей**

# %%
def quarter_theory(df, levels=2):
    """
    Улучшенная версия индикатора четвертей с автоматической проверкой даты.
    
    Параметры:
        df (pd.DataFrame): DataFrame с колонкой 'Data' (строка или datetime)
        levels (int): Глубина разбиения (1=4 четверти, 2=16, 3=64 и т.д.)
    
    Возвращает:
        pd.DataFrame: Копия исходного DF с добавленными колонками Q_L{level}_{quarter}
        
    Исключения:
        ValueError: Если колонка 'Date' отсутствует или не может быть преобразована
    """
    # Создаем копию, чтобы не менять исходный DF
    result_df = df.copy()
    
    # Проверка наличия колонки
    if 'Data' not in result_df.columns:
        raise ValueError("DataFrame должен содержать колонку 'Data'")
    
    # Преобразование и проверка типа даты
    try:
        if not pd.api.types.is_datetime64_any_dtype(result_df['Data']):
            result_df['Data'] = pd.to_datetime(result_df['Data'], errors='raise')
    except Exception as e:
        raise ValueError(f"Ошибка преобразования даты: {str(e)}")
    
    # Оптимизированное вычисление времени
    dt = result_df['Data'].dt
    time_in_sec = dt.hour * 3600 + dt.minute * 60 + dt.second
    normalized_time = time_in_sec / 86400  # Нормализуем до [0, 1)
    
    # Предварительное вычисление всех индикаторов
    quarter_cols = {}
    for level in range(1, levels + 1):
        num_quarters = 4 ** level
        quarter_bins = np.arange(num_quarters + 1) / num_quarters
        quarters = np.digitize(normalized_time, quarter_bins, right=False) - 1
        
        for q in range(num_quarters):
            quarter_cols[f'Q_L{level}_{q+1}'] = (quarters == q).astype(np.int8)
    
    # Добавляем все колонки за одну операцию
    result_df = pd.concat([result_df, pd.DataFrame(quarter_cols)], axis=1)
    
    return result_df


# %% [markdown]
# ML индикаторы

# %%
@njit
def _compute_tau(prices, lags):
    tau = np.empty(len(lags))
    for i in range(len(lags)):
        lag = lags[i]
        if lag >= len(prices):
            tau[i] = np.nan
        else:
            diffs = prices[lag:] - prices[:-lag]
            tau[i] = np.std(diffs)
    return tau

def hurst_exponent(prices, max_lag=100, poly_deg=1):
    prices = np.asarray(prices)
    if len(prices) < 20:  # абсолютный минимум
        return np.nan

    effective_max_lag = min(max_lag, len(prices) - 1)
    lags = np.arange(2, effective_max_lag)

    tau = _compute_tau(prices, lags)
    # фильтруем недопустимые значения
    valid = ~np.isnan(tau) & (tau > 0)
    if valid.sum() < poly_deg + 1:
        return np.nan

    try:
        hurst = np.polyfit(np.log(lags[valid]), np.log(tau[valid]), deg=poly_deg)[0]
        return hurst
    except Exception:
        return np.nan

@njit
def sample_entropy(close_prices, m=2, r=0.2):
    n = len(close_prices)
    if n <= m:
        return 0.0
    std = np.std(close_prices)
    if std == 0:
        return 0.0
    r *= std
    
    count = 0
    patterns = np.lib.stride_tricks.sliding_window_view(close_prices, m)
    
    for i in range(len(patterns)):
        for j in range(i+1, len(patterns)):
            if np.max(np.abs(patterns[i] - patterns[j])) <= r:
                count += 1
                
    return -np.log(count / (n - m)) if count > 0 else 0.0

def dominant_frequency(close_prices):
    n = len(close_prices)
    yf = rfft(close_prices - np.mean(close_prices))
    xf = rfftfreq(n, 1)
    return xf[np.argmax(np.abs(yf))]

def nar_residual(close_prices, lag=5):
    X = np.array([close_prices[i-lag:i] for i in range(lag, len(close_prices))])
    y = close_prices[lag:]
    model = Ridge(alpha=1.0).fit(X, y)
    return (y - model.predict(X))[-1] if len(y) > 0 else np.nan

def wasserstein_distance(prices_window, window=100, compare=50):
    if len(prices_window) < window:
        return np.nan
    current = prices_window[-compare:]
    reference = prices_window[-window:-compare]
    return ot.wasserstein_1d(current, reference)



# %% [markdown]
# # Keltner_func

# %%
def Keltner_func(df, 
                 base_atr_period=40, 
                 base_multiplier=2,
                 alt_atr_period=100,
                 alt_multiplier=1.5,
                 alt_ema_shift=5):
    
    df = df.copy()
    eps = 1e-8
    
    # 1. Базовый Keltner Channel Width
    def _base_keltner(h, l, c, atr_period, multiplier):
        typical_price = (h + l + c) / 3
        ema = typical_price.ewm(span=atr_period, adjust=False).mean()
        
        tr = pd.concat([
            h - l,
            (h - c.shift()).abs(),
            (l - c.shift()).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.ewm(span=atr_period, adjust=False).mean()
        return (2 * multiplier * atr) / (ema + eps)
    
    # 2. Альтернативный Keltner Width
    def _alt_keltner(h, l, c, atr_period, multiplier, shift):
        typical_price = (h + l + c) / 3
        ema = typical_price.shift(shift).ewm(span=atr_period, adjust=False).mean()
        
        tr = pd.concat([
            h - l,
            (h - c.shift()).abs(),
            (l - c.shift()).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.ewm(span=atr_period * 2, adjust=False).mean()
        return (multiplier * atr) / (ema + eps)

    # Расчет быстрой версии Keltner для нормализации
    df['KeltnerWidth_5'] = _base_keltner(df['High'], df['Low'], df['Close'], 5, 1.0)
    df['KeltnerWidth_5'] = df['KeltnerWidth_5'].replace([np.inf, -np.inf], np.nan).ffill()
    keltner5 = df['KeltnerWidth_5']  # Series для удобства

    # Вычисляем основные Keltner Width
    df['KeltnerWidth_v2'] = _base_keltner(df['High'], df['Low'], df['Close'],
                                          base_atr_period, base_multiplier)
    
    df['KeltnerWidth_v3'] = _alt_keltner(df['High'], df['Low'], df['Close'],
                                         alt_atr_period, alt_multiplier, alt_ema_shift)
    
    # Альтернативная нормализация
    for col in ['KeltnerWidth_v2', 'KeltnerWidth_v3']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).ffill()
        df[f'{col}_norm'] = (df[col] - keltner5) / (df[col] + eps)
    df.drop(columns=['KeltnerWidth_5', 'KeltnerWidth_v2', 'KeltnerWidth_v3'     ], inplace=True)

    return df


# %% [markdown]
# **Супер Тренд**

# %%
def super_trend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0, eps: float = 1e-8) -> pd.DataFrame:
    """
    Расчёт индикатора SuperTrend с нормализованным выходом [-1, 1].
    Добавляет один столбец: super_trend_{period}_{multiplier} ∈ [-1, 1]
    где 1 - сильный бычий сигнал, -1 - сильный медвежий
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None")

    high = df['High']
    low = df['Low']
    close = df['Close']

    hl2 = (high + low) / 2.0

    # True Range с защитой от NaN
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(0)

    # ATR (SMA) — только прошлое
    atr = tr.rolling(period).mean().shift(1)

    # Upper/Lower Bands
    upper_band = (hl2 + multiplier * atr).shift(1)
    lower_band = (hl2 - multiplier * atr).shift(1)

    # Инициализация
    supertrend = pd.Series(index=df.index, dtype='float64')
    trend_direction = pd.Series(index=df.index, dtype='float64')  # 1 или -1

    for i in range(len(df)):
        if i < period:
            supertrend.iloc[i] = np.nan
            trend_direction.iloc[i] = 0
        elif i == period:
            if close.iloc[i] < upper_band.iloc[i]:
                supertrend.iloc[i] = upper_band.iloc[i]
                trend_direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = lower_band.iloc[i]
                trend_direction.iloc[i] = 1
        else:
            prev_st = supertrend.iloc[i-1]
            if trend_direction.iloc[i-1] > 0:
                new_st = max(lower_band.iloc[i], prev_st)
                if close.iloc[i] < new_st:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    trend_direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = new_st
                    trend_direction.iloc[i] = 1
            else:
                new_st = min(upper_band.iloc[i], prev_st)
                if close.iloc[i] > new_st:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    trend_direction.iloc[i] = 1
                else:
                    supertrend.iloc[i] = new_st
                    trend_direction.iloc[i] = -1

    # Нормализация сигнала с учетом направления и расстояния
    normalized_signal = trend_direction * (1 - (atr / (abs(close - supertrend) + eps)).clip(0, 1))
    normalized_signal = normalized_signal.fillna(0).clip(-1, 1)

    # Добавляем столбец с уникальным именем, зависящим от параметров
    df[f'super_trend_{period}_{multiplier}'] = normalized_signal

    return df


# %% [markdown]
# **fibo_dinamic**

# %%
def fibo_dinamic(df: pd.DataFrame, period: int = 300, eps: float = 1e-8) -> pd.DataFrame:
    """
    Добавляет индикатор fibo_dinamic_<period>, нормализованный от -1 до 1.
    Отражает положение Close относительно середины диапазона последних <period> свечей.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is None or empty")
        
    high_roll = df['High'].rolling(window=period)
    low_roll = df['Low'].rolling(window=period)
    
    max_price = high_roll.max()
    min_price = low_roll.min()
    range_price = max_price - min_price
    
    midpoint = min_price + 0.5 * range_price  # уровень 0.5 фибо
    
    relative_position = (df['Close'] - midpoint) / (range_price + eps)
    relative_position = relative_position.clip(-1, 1)  # ограничиваем диапазон
    
    col_name = f'fibo_dinamic_{period}'
    df[col_name] = relative_position

    return df


# %% [markdown]
# **Deep learning индикаторы**

# %%

# Параметры по умолчанию
EPS = 1e-8

def spectral_entropy(signal, window_size=100):
    """Вычисляет спектральную энтропию для последнего окна"""
    if len(signal) < window_size:
        return np.nan

    window = signal[-window_size:]
    f, Pxx = welch(window - np.mean(window), nperseg=len(window))
    Pxx = Pxx / (np.sum(Pxx) + EPS)
    entropy = -np.sum(Pxx * np.log2(Pxx + EPS))
    return entropy / np.log2(len(Pxx))  # нормализация

def trend_stability(close, window=100):
    """Считает устойчивость тренда: сколько раз сменился знак"""
    if len(close) < window:
        return np.nan
    diffs = np.diff(np.sign(np.diff(close[-window:])))
    changes = np.sum(diffs != 0)
    return 1 - changes / (window - 2 + EPS)  # нормализуем: меньше смен — выше стабильность

def higuchi_fd(signal, k_max=5):
    """Быстрая аппроксимация фрактальной размерности методом Хигучи"""
    if len(signal) < k_max + 1:
        return np.nan
    L = []
    x = signal - np.mean(signal)
    N = len(x)
    for k in range(1, k_max + 1):
        Lk = np.mean([
            np.sum(np.abs(np.diff(x[m::k]))) * (N - 1) / (((N - m) // k) * k + EPS)
            for m in range(k)
        ])
        L.append(Lk)
    lnL = np.log(L)
    lnk = np.log(np.arange(1, k_max + 1))
    if np.any(np.isnan(lnL)):
        return np.nan
    coeffs = np.polyfit(lnk, lnL, 1)
    return coeffs[0]  # результат обычно от 1 до 2

# Обёртка для DataFrame
def add_structural_features(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    closes = df['Close'].values

    spectral = []
    stability = []
    fractal = []

    for i in range(len(df)):
        window_data = closes[max(0, i - window + 1):i + 1]
        spectral.append(spectral_entropy(window_data, window))
        stability.append(trend_stability(window_data, window))
        fractal.append(higuchi_fd(window_data, k_max=5))

    df[f'spectral_entropy_{window}'] = spectral
    df[f'trend_stability_{window}'] = stability
    df[f'fractal_dim_{window}'] = fractal

    # нормализация признаков к [0, 1] с защитой от деления на 0
    se = df[f'spectral_entropy_{window}']
    ts = df[f'trend_stability_{window}']
    fd = df[f'fractal_dim_{window}']

    # Фрактальная размерность [1, 2] → [0, 1]
    fd_norm = (fd - 1.0) / (2.0 - 1.0 + EPS)

    df[f'signal_complexity_{window}'] = se * ts * fd_norm

    return df


# %% [markdown]
# **MACD**

# %%
def add_macd(df: pd.DataFrame, fast=12, slow=26, signal=9, window_norm=100, eps=1e-8) -> pd.Series:
    """
    Возвращает нормализованный MACD (гистограмма) в виде Series.
    Назначай его в df: df['macd_f_12_s26'] = add_macd(df, fast=12, slow=26)

    - fast: период быстрой EMA
    - slow: период медленной EMA
    - signal: период сигнальной линии
    - window_norm: окно для нормализации гистограммы
    """
    if df is None or df.empty or 'Close' not in df.columns:
        raise ValueError("DataFrame пустой или не содержит колонку 'Close'")

    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - signal_line

    # Нормализация гистограммы (от -1 до 1 на скользящем окне)
    rolling_max = macd_hist.rolling(window=window_norm).max()
    rolling_min = macd_hist.rolling(window=window_norm).min()
    macd_rel = 2 * (macd_hist - rolling_min) / (rolling_max - rolling_min + eps) - 1

    return macd_rel


# %% [markdown]
# # Кастомные индикаторы

# %% [markdown]
# ## Уровень сопротивления

# %%
def calculate_resistance_distance(df: pd.DataFrame,
                                  long_window: int = 200,
                                  short_window: int = 20,
                                  segment_size: int = 20,
                                  price_col: str = 'Close',
                                  ema_span: int = 9,
                                  eps: float = 1e-8,
                                  debug: bool = False) -> pd.Series:
    """
    Индикатор расстояния до нисходящей наклонной линии сопротивления, построенной от глобального максимума.
    """
    try:
        if len(df) < long_window:
            return pd.Series(1, index=df.index, name='resistance_slope_dist')

        highs = df['High'].values
        closes = df[price_col].values
        ema9 = pd.Series(closes).ewm(span=ema_span, adjust=False).mean().values
        result = np.full(len(df), 1.0, dtype=np.float32)

        for i in range(long_window, len(df)):
            window_highs = highs[i - long_window:i]
            recent_highs = highs[i - short_window:i]

            max_200 = np.max(window_highs)
            max_20 = np.max(recent_highs)

            if max_200 <= max_20:
                continue  # нет нисходящего тренда

            # Индекс глобального максимума в окне
            global_max_idx = np.argmax(window_highs)
            if global_max_idx >= long_window - segment_size:
                continue  # слишком мало места после пика для сегментов

            # Область после глобального максимума
            segment_start = global_max_idx + 1
            remaining = long_window - segment_start
            num_segments = remaining // segment_size
            if num_segments < 2:
                continue

            segment_highs = window_highs[segment_start:segment_start + num_segments * segment_size]
            segment_highs_reshaped = segment_highs.reshape(num_segments, segment_size)
            segment_maxima = np.nanmax(segment_highs_reshaped, axis=1)

            if np.any(np.isnan(segment_maxima)):
                continue

            # Включаем сам пик как начальную точку
            y_points = np.concatenate([[max_200], segment_maxima])
            x_points = np.arange(len(y_points)).reshape(-1, 1)

            # Линейная регрессия
            model = LinearRegression()
            model.fit(x_points, y_points)
            predicted_resistance = model.predict([[len(y_points) - 1]])[0]

            current_ema = ema9[i]
            distance = (current_ema - predicted_resistance) / (abs(predicted_resistance) + eps)
            result[i] = np.clip(distance, -1, 1)

        return pd.Series(result, index=df.index, name='resistance_slope_dist')

    except Exception as e:
        if debug:
            print(f"⚠️ Resistance indicator error: {e}")
        return pd.Series(1, index=df.index, name='resistance_slope_dist')

# df['resistance_slope_dist'] = calculate_resistance_distance(df)


# %% [markdown]
# ## Уровень поддержки

# %%
def calculate_support_distance(
    df: pd.DataFrame,
    long_window: int = 200,
    short_window: int = 20,
    segment_size: int = 20,
    price_col: str = 'Close',
    ema_span: int = 9,
    eps: float = 1e-8,
    debug: bool = False
) -> pd.Series:
    """
    Индикатор расстояния до линии восходящей поддержки, построенной от глобального минимума.
    """
    try:
        if len(df) < long_window:
            return pd.Series(-1, index=df.index, name='support_slope_dist')

        lows = df['Low'].values
        closes = df[price_col].values
        ema9 = pd.Series(closes).ewm(span=ema_span, adjust=False).mean().values
        result = np.full(len(df), -1.0, dtype=np.float32)

        for i in range(long_window, len(df)):
            window_lows = lows[i - long_window:i]
            recent_lows = lows[i - short_window:i]

            min_200 = np.min(window_lows)
            min_20 = np.min(recent_lows)

            # Условие на восходящую поддержку
            if min_200 >= min_20:
                continue

            # Индекс глобального минимума в окне
            global_min_idx = np.argmin(window_lows)
            if global_min_idx >= long_window - segment_size:
                continue  # слишком мало места после минимума для сегментов

            # Область после глобального минимума
            segment_start = global_min_idx + 1
            remaining = long_window - segment_start
            num_segments = remaining // segment_size
            if num_segments < 2:
                continue

            segment_lows = window_lows[segment_start:segment_start + num_segments * segment_size]
            segment_lows_reshaped = segment_lows.reshape(num_segments, segment_size)
            segment_minima = np.nanmin(segment_lows_reshaped, axis=1)

            if np.any(np.isnan(segment_minima)):
                continue

            # Включаем сам минимум как первую точку
            y_points = np.concatenate([[min_200], segment_minima])
            x_points = np.arange(len(y_points)).reshape(-1, 1)

            # Линейная регрессия
            model = LinearRegression()
            model.fit(x_points, y_points)
            predicted_support = model.predict([[len(y_points) - 1]])[0]

            current_ema = ema9[i]
            distance = (current_ema - predicted_support) / (abs(predicted_support) + eps)
            result[i] = np.clip(distance, -1, 1)

        return pd.Series(result, index=df.index, name='support_slope_dist')

    except Exception as e:
        if debug:
            print(f"⚠️ Support indicator error: {e}")
        return pd.Series(-1, index=df.index, name='support_slope_dist')


# %% [markdown]
# # Кастомные индикаторы объема

# %%
def add_volume_dynamics(df: pd.DataFrame, col: str = 'Volume', spike_sigma: float = 1.5) -> pd.DataFrame:
    """
    Добавляет производные признаки от объема:
    - rolling std, diff, percent change, 2-я производная
    - volume spikes (всплески выше среднего + x * std)
    - кластеры вверх/вниз
    - нормализация [-1, 1] всех новых признаков
    """
    df = df.copy()

    # Стандартное отклонение
    df[f'{col}_rolling_std_10'] = df[col].rolling(window=10).std()
    df[f'{col}_rolling_std_30'] = df[col].rolling(window=30).std()

    # Первая производная
    df[f'{col}_diff_1'] = df[col].diff()
    df[f'{col}_rolling_diff_mean_10'] = df[col].diff().rolling(window=10).mean()

    # Процентное изменение
    df[f'{col}_pct_change'] = df[col].pct_change()
    df[f'{col}_pct_change_rolling_mean_10'] = df[col].pct_change().rolling(window=10).mean()

    # Вторая производная
    df[f'{col}_diff2'] = df[col].diff().diff()

    # Volume spikes (объем выше среднего + x * std)
    vol_mean = df[col].rolling(window=20).mean()
    vol_std = df[col].rolling(window=20).std()
    spike_threshold = vol_mean + spike_sigma * vol_std
    df[f'{col}_spike'] = (df[col] > spike_threshold).astype(int)
    df[f'{col}_spike'] = df[f'{col}_spike'].rolling(window=10).sum()  # сумма спайков за окно

    # Кластеры объема вверх/вниз
    df['direction'] = np.sign(df['Close'] - df['Open'])  # +1 = вверх, -1 = вниз, 0 = без изменения
    df[f'{col}_up_volume'] = df[col] * (df['direction'] > 0)
    df[f'{col}_down_volume'] = df[col] * (df['direction'] < 0)

    df[f'{col}_up_cluster'] = df[f'{col}_up_volume'].rolling(window=10).sum()
    df[f'{col}_down_cluster'] = df[f'{col}_down_volume'].rolling(window=10).sum()

    # Разность кластеров (направленная активность)
    df[f'{col}_volume_cluster_diff'] = df[f'{col}_up_cluster'] - df[f'{col}_down_cluster']

    # Удаляем промежуточные колонки
    df.drop(columns=['direction', f'{col}_up_volume', f'{col}_down_volume'], inplace=True)

    # Нормализация всех новых признаков (robust min-max)
    for colname in df.columns:
        if colname.startswith(col + '_') and not df[colname].isna().all():
            rolling_min = df[colname].rolling(window=100).min()
            rolling_max = df[colname].rolling(window=100).max()
            df[colname] = 2 * (df[colname] - rolling_min) / (rolling_max - rolling_min + 1e-8) - 1

    return df


# %%
def add_volume_features(df: pd.DataFrame, window: int = 20, eps: float = 1e-9) -> pd.DataFrame:
    """
    Добавляет признаки:
    - Volume_skew: асимметрия (скошенность) объема
    - Volume_volatility: стандартное отклонение объема (нормализованное)
    - Volume_spike: на сколько текущий объем превышает медиану (от 0 до 1)

    Защита от деления на 0 через eps. Без подглядываний (все lagged).
    """
    vol = df['Volume']

    # Скользящие метрики
    rolling_median = vol.rolling(window).median()
    rolling_mean = vol.rolling(window).mean()
    rolling_std = vol.rolling(window).std()
    rolling_skew = vol.rolling(window).skew()

    # Volume_skew (асимметрия)
    df[f'volume_skew_{window}'] = rolling_skew.clip(-5, 5) / 5  # нормализуем в диапазон -1..1

    # Volume_volatility (волатильность объема)
    df[f'volume_volatility_{window}'] = (rolling_std / (rolling_mean + eps)).clip(0, 5) / 5

    # Volume_spike (от 0 до 1 — насколько выше медианы)
    spike_score = (vol - rolling_median) / (rolling_median + eps)
    df[f'volume_spike_{window}'] = spike_score.clip(0, 3) / 3  # только вверх, нормализуем

    return df


# %% [markdown]
# **Кастомные индикаторы объема и размера свеч**

# %%
def add_candle_vol_size_features(df, window=20, eps=1e-8):
    """
    Добавляет к df признаки, отражающие взаимосвязь свечных размеров и объема:
    - candle_vol_size_prod: (body * full) / volume
    - candle_vol_size_hmean: ((body * full) / (body + full)) / volume
    Также добавляет версии:
    - _tanh: ограничено от -1 до 1
    - _log: логарифм от абсолютного значения
    """
    body = (df['Close'] - df['Open']).abs()
    full = df['High'] - df['Low']

    size_prod = body * full
    size_hmean = (body * full) / (body + full + eps)

    # Нормализация по среднему объёму
    avg_volume = df['Volume'].rolling(window).mean()
    med_volume = df['Volume'].rolling(window).median()
    V = df['Volume'] / (avg_volume + med_volume + eps)

    size_prod_roll = size_prod / (size_prod.rolling(window).mean() + eps)
    size_hmean_roll = size_hmean / (size_hmean.rolling(window).mean() + eps)

    # Основные признаки
    prod = size_prod_roll / (V + eps)
    hmean = size_hmean_roll / (V + eps)

    # Добавляем признаки в DataFrame
    df['candle_vol_size_prod'] = prod
    df['candle_vol_size_prod_tanh'] = np.tanh(prod)
    df['candle_vol_size_prod_log'] = np.sign(prod) * np.log1p(np.abs(prod))

    df['candle_vol_size_hmean'] = hmean
    df['candle_vol_size_hmean_tanh'] = np.tanh(hmean)
    df['candle_vol_size_hmean_log'] = np.sign(hmean) * np.log1p(np.abs(hmean))

    return df


# %% [markdown]
# # add_choppiness_index

# %%
def add_choppiness_index(df, period=14, eps=1e-8):
    """
    Добавляет столбец 'choppiness_index_{period}' в DataFrame df.
    Значения нормализованы от 0 до 1.
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_sum = true_range.rolling(period).sum()
    high_max = high.rolling(period).max()
    low_min = low.rolling(period).min()
    range_max_min = high_max - low_min

    # CHOP = 100 * log10(ATR_sum / range) / log10(period)
    chop = 100 * np.log10((atr_sum + eps) / (range_max_min + eps)) / np.log10(period + eps)

    # Нормализация от 0 до 1 (по теоретическому диапазону от ~20 до ~61.8)
    chop_norm = (chop - 20) / (61.8 - 20)
    df[f'choppiness_index_{period}'] = chop_norm.clip(0, 1)

    return df


# %% [markdown]
# # Bollinger

# %%
def add_bollinger_features(df, period=20, std_mult=2, eps=1e-8):
    """
    Добавляет признаки по Bollinger Bands:
    - расстояние от цены до средней
    - нормализованная ширина полос
    - z-оценка положения цены в канале
    """
    price = df['Close']
    sma = price.rolling(window=period).mean()
    std = price.rolling(window=period).std()

    upper = sma + std_mult * std
    lower = sma - std_mult * std

    # Расстояние до средней, нормализованное
    df[f'bb_z_{period}'] = (price - sma) / (std + eps)

    # Нормализованная ширина канала
    df[f'bb_width_{period}'] = (upper - lower) / (sma + eps)

    return df


# %% [markdown]
# bollinger_awesome_alert

# %%
def bollinger_awesome_alert(df, bb_use_ema=False, bb_filter=False, sqz_filter=False, 
                           bb_length=20, bb_mult=2.0, fast_ma_len=3, 
                           nLengthSlow=34, nLengthFast=5, sqz_length=100, 
                           sqz_threshold=50):
    """
    Реализация Bollinger Awesome Alert R1.1 by JustUncleL.
    Только добавляет новые колонки, ничего не удаляет.
    Убраны коррелирующие признаки (оставлен один из каждой коррелирующей пары).
    
    Параметры:
    - df: DataFrame с колонками ['Close', 'Open', 'Low', 'High', 'Volume']
    - bb_use_ema: использовать EMA вместо SMA для Bollinger Bands (по умолчанию False)
    - bb_filter: фильтровать сигналы по Bollinger Bands (по умолчанию False)
    - sqz_filter: фильтровать сигналы по "сжатию" Bollinger Bands (по умолчанию False)
    - остальные параметры соответствуют оригинальному индикатору
    
    Возвращает:
    - Тот же DataFrame с добавленными колонками индикатора (без коррелирующих признаков)
    """
    
    # Создаем копию DataFrame чтобы избежать предупреждений
    df = df.copy()
    
    # 1. Bollinger Bands
    df['bb_basis'] = df['Close'].ewm(span=bb_length, adjust=False).mean() if bb_use_ema else \
                     df['Close'].rolling(window=bb_length).mean()
    
    # Убраны bb_dev, bb_upper, bb_lower (коррелируют с bb_basis и между собой)
    
    # 2. Быстрая EMA
    df['fast_ma'] = df['Close'].ewm(span=fast_ma_len, adjust=False).mean()
    
    # 3. Awesome Oscillator
    hl2 = (df['High'] + df['Low']) / 2
    # Убраны xSMA1_hl2 и xSMA2_hl2 (коррелируют с fast_ma и bb_basis)
    df['xSMA1_SMA2'] = hl2.rolling(window=nLengthFast).mean() - hl2.rolling(window=nLengthSlow).mean()
    
    # Направление AO
    df['AO'] = np.where(df['xSMA1_SMA2'] >= 0, 
                       np.where(df['xSMA1_SMA2'] > df['xSMA1_SMA2'].shift(1), 1, 2),
                       np.where(df['xSMA1_SMA2'] > df['xSMA1_SMA2'].shift(1), -1, -2))
    
    # 4. Сжатие Bollinger Bands
    # Убраны spread, avgspread, bb_squeeze (используется только bb_squeeze в фильтре)
    if sqz_filter:
        spread = 2 * df['Close'].rolling(window=bb_length).std() * bb_mult  # bb_upper - bb_lower = 2*bb_dev
        avgspread = spread.rolling(window=sqz_length).mean()
        df['bb_squeeze'] = spread / avgspread * 100
    
    # 5. ATR (для полноты, хотя в сигналах не используется)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    # Убран bb_offset (коррелирует с atr)
    
    # 6. Генерация сигналов
    df['BA_Signal'] = 0
    
    # Вычисляем bb_upper и bb_lower временно, если нужно для фильтра
    if bb_filter or sqz_filter:
        bb_dev = df['Close'].rolling(window=bb_length).std() * bb_mult
        bb_upper = df['bb_basis'] + bb_dev
        bb_lower = df['bb_basis'] - bb_dev
    
    # Условия для BUY
    buy_cond = (
        (df['fast_ma'] > df['bb_basis']) & 
        (df['fast_ma'].shift(1) <= df['bb_basis'].shift(1)) & 
        (df['Close'] > df['bb_basis']) & 
        (np.abs(df['AO']) == 1) & 
        ((not bb_filter) | (df['Close'] < bb_upper)) & 
        ((not sqz_filter) | (df['bb_squeeze'] > sqz_threshold))
    )
    
    # Условия для SELL
    sell_cond = (
        (df['fast_ma'] < df['bb_basis']) & 
        (df['fast_ma'].shift(1) >= df['bb_basis'].shift(1)) & 
        (df['Close'] < df['bb_basis']) & 
        (np.abs(df['AO']) == 2) & 
        ((not bb_filter) | (df['Close'] > bb_lower)) & 
        ((not sqz_filter) | (df['bb_squeeze'] > sqz_threshold))
    )
    
    df.loc[buy_cond, 'BA_Signal'] = 1    # BUY сигнал
    df.loc[sell_cond, 'BA_Signal'] = -1   # SELL сигнал
    
    return df


# %% [markdown]
# # parabolic_sar

# %%
def add_parabolic_sar_feature(df, step=0.02, max_step=0.2):
    """
    Добавляет нормализованный признак на основе индикатора Parabolic SAR.
    Значения нормализованы относительно High и Low текущей свечи.
    Без подглядывания в будущее.
    """
    eps = 1e-8
    
    psar_values = psar(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        step=step,
        max_step=max_step
    )['PSARl_'+str(step)+'_'+str(max_step)]  # Получаем только значения PSAR
    
    # Нормализация: где находится PSAR по отношению к High/Low
    df['parabolic_sar_norm'] = (psar_values - df['Low']) / (df['High'] - df['Low'] + eps)
    
    return df


# %% [markdown]
# # Индикатор волн

# %%
def add_wave_phase_position(df, lookback=700, smooth_window=21, price_col='Close'):
    # Добавляем динамическое окно сглаживания
    smooth_window = min(smooth_window, lookback//10)
    
    # Модифицированный Z-score (менее чувствительный к выбросам)
    median = df[price_col].rolling(lookback).median()
    mad = 1.4826 * df[price_col].rolling(lookback).apply(lambda x: np.median(np.abs(x - np.median(x))))
    z = (df[price_col] - median) / (mad + 1e-8)
    z = z.clip(-4,4)/4  # Более мягкое ограничение
    
    # Двойное сглаживание
    ewma_fast = z.ewm(span=smooth_window//3).mean()
    ewma_slow = z.ewm(span=smooth_window).mean()
    
    # Расхождение как дополнительный признак
    divergence = ewma_fast - ewma_slow
    
    df['wave_phase'] = ewma_slow
    df['phase_divergence'] = divergence
    
    return df


# %%
def add_fft_phase_position(df, price_col='Close', window=700):
    """
    Добавляет колонку 'fft_phase_position' — фазу главной синусоиды, от -1 до 1.
    Используется FFT на последних `window` точках.
    """
    prices = df[price_col].values
    fft_phase = np.full(len(prices), np.nan)  # результат
    
    for i in range(window, len(prices)):
        segment = prices[i - window:i]
        segment = segment - np.mean(segment)  # убираем DC-смещение
        
        fft = np.fft.fft(segment)
        freqs = np.fft.fftfreq(window)
        
        # Берем только положительные частоты, исключая DC
        pos_freqs = freqs[1:window // 2]
        magnitudes = np.abs(fft[1:window // 2])
        
        if len(magnitudes) == 0:
            continue
        
        dominant_idx = np.argmax(magnitudes)
        phase = np.angle(fft[dominant_idx + 1])  # +1 из-за исключения DC
        
        # Нормируем фазу: от -π до π => от -1 до 1
        phase_norm = phase / np.pi
        fft_phase[i] = phase_norm

    df['fft_phase_position'] = fft_phase
    return df


# %%
def adaptive_zscore(df, lookback=700, vol_span=100, price_col='Close'):
    median = df[price_col].rolling(lookback).median()
    ewma = df[price_col].ewm(span=vol_span)
    ewma_std = (ewma.var())**0.5
    z = (df[price_col] - median) / (ewma_std + 1e-8)
    df['adaptive_zscore'] = z
    return df


# %%
def quantile_position(df, lookback=700, price_col='Close'):
    """
    Позиция цены в историческом распределении (percentile rank).
    Возвращает значения от -1 (минимум за период) до 1 (максимум).
    """
    rolling_rank = df[price_col].rolling(lookback).rank(pct=True)
    df['quantile_position'] = 2 * (rolling_rank - 0.5)  # преобразуем в [-1, 1]
    return df


# %%
def atr_position(df, lookback=700, atr_window=14, price_col='Close'):
    # Расчет True Range
    hl = df['High'] - df['Low']
    hc = abs(df['High'] - df[price_col].shift(1))
    lc = abs(df['Low'] - df[price_col].shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    
    atr = tr.rolling(atr_window).mean()
    price_ma = df[price_col].rolling(lookback).mean()
    pos = (df[price_col] - price_ma) / (2 * atr + 1e-8)
    df['atr_position'] = pos.clip(-1, 1)
    return df


# %%
def rsi_stochastic_hybrid(df, rsi_window=14, stoch_window=14, smooth=3, price_col='Close'):
    """
    Гибрид RSI и Stochastic Oscillator.
    Возвращает значения от -1 до 1.
    """
    # RSI часть
    delta = df[price_col].diff()
    gain = delta.clip(lower=0).rolling(rsi_window).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_window).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic часть
    high = df['High'].rolling(stoch_window).max()
    low = df['Low'].rolling(stoch_window).min()
    stoch = 100 * (df[price_col] - low) / (high - low + 1e-8)
    stoch_smooth = stoch.rolling(smooth).mean()
    
    # Комбинация (нормализованная)
    hybrid = 0.5*(rsi/50 - 1) + 0.5*(stoch_smooth/50 - 1)
    df['rsi_stoch_hybrid'] = hybrid.clip(-1, 1)
    return df


# %% [markdown]
# # Нормализованные Close

# %%
def add_close_window_norm_pca(df: pd.DataFrame, window: int = 700, n_components: int = 5) -> pd.DataFrame:
    """
    Добавляет PCA-признаки по нормализованному окну Close за последние window свечей.
    Возвращает df с новыми колонками: close_pca_0, ..., close_pca_{n_components-1}
    """
    df = df.copy()
    close = df['Close'].values
    features = []

    for i in range(window, len(close)):
        segment = close[i - window:i]
        # Нормализация от -1 до 1
        scaled = 2 * (segment - np.min(segment)) / (np.max(segment) - np.min(segment) + 1e-8) - 1
        features.append(scaled)

    # Преобразуем в матрицу
    X = np.array(features)
    
    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Названия колонок
    col_names = [f'close_pca_{i}' for i in range(n_components)]

    # Добавим NaN в начало (до window строк), чтобы длины совпадали
    pca_df = pd.DataFrame(np.full((len(df), n_components), np.nan), columns=col_names)
    pca_df.iloc[window:] = X_pca

    # Добавляем к исходному df
    df = pd.concat([df, pca_df], axis=1)
    return df


# %% [markdown]
# Колонки для CNN модели

# %%
def add_close_window_columns(df, window=60, price_col='Close'):
    """
    Безопасное создание оконных признаков без утечки данных
    с нормализацией внутри каждого окна.
    """
    close = df[price_col].values
    
    # Создаём пустой массив для результатов
    windowed = np.full((len(df), window), np.nan)
    
    for i in range(window, len(df)):
        window_data = close[i-window:i]
        
        # Нормализация внутри окна (от -1 до 1)
        min_val = window_data.min()
        max_val = window_data.max()
        normalized = 2 * ((window_data - min_val) / (max_val - min_val + 1e-8)) - 1
        
        windowed[i] = normalized
    
    # Добавляем колонки в DataFrame
    for i in range(window):
        df[f'close_win_{i}'] = windowed[:, i]
    
    return df.iloc[window:]  # Удаляем начальные NaN


# %% [markdown]
# # TradingView индикаторы

# %% [markdown]
# ## PPO

# %%
def ppo(df, 
        source_col='Close',  # По умолчанию используем Close для избежания look-ahead
        long_term_div=True,
        div_lookback_period=55,
        fastLength=12,
        slowLength=26,
        signalLength=9,
        smoother=2,
        signal_duration=3):  # Продолжительность сигнала в свечах
    
    # Создаем копию DataFrame чтобы не изменять исходный
    df = df.copy()
    
    # Рассчитываем индикаторы (только по закрытым свечам)
    fastMA = df[source_col].ewm(span=fastLength, adjust=False).mean()
    slowMA = df[source_col].ewm(span=slowLength, adjust=False).mean()
    macd = fastMA - slowMA
    macd2 = (macd / slowMA) * 100
    df['PPO'] = macd2.rolling(window=smoother).mean()
    
    # Функция для нахождения предыдущих экстремумов без look-ahead
    def valuewhen(condition, series, occurrence):
        # Создаем копию series с NaN где условие False
        masked = series.where(condition)
        # Заполняем вперед и сдвигаем
        return masked.ffill().shift(occurrence + 1)  # +1 чтобы избежать look-ahead
    
    # Идентификация минимумов PPO
    oscMins = (df['PPO'] > df['PPO'].shift(1)) & (df['PPO'].shift(1) < df['PPO'].shift(2))
    
    # Идентификация минимумов цены (по закрытым свечам)
    low = df['Low']
    cond1 = (low > low.shift(1)) & (low.shift(1) < low.shift(2))
    cond2 = (low.shift(1) == low.shift(2)) & (low.shift(1) < low) & (low.shift(1) < low.shift(3))
    cond3 = (low.shift(1) == low.shift(2)) & (low.shift(1) == low.shift(3)) & (low.shift(1) < low) & (low.shift(1) < low.shift(4))
    cond4 = (low.shift(1) == low.shift(2)) & (low.shift(1) == low.shift(3)) & (low.shift(1) == low.shift(4)) & (low.shift(1) < low) & (low.shift(1) < low.shift(5))
    
    priceMins = cond1 | cond2 | cond3 | cond4
    
    # Находим экстремумы
    current_ppo_bottom = valuewhen(oscMins, df['PPO'].shift(1), 0)
    prev_ppo_bottom = valuewhen(oscMins, df['PPO'].shift(1), 1)
    current_price_bottom = valuewhen(priceMins, df['Low'].shift(1), 0)
    
    # Рассчитываем y6 (фильтр для минимумов цены)
    def rolling_min_with_offset(s, window):
        return s.rolling(window, min_periods=1).min().shift(1)  # Сдвиг для избежания look-ahead
    
    y6 = valuewhen(oscMins, 
                  priceMins.rolling(5, min_periods=1).apply(
                      lambda x: x.iloc[:-1][::-1].argmax() if x[:-1].any() else np.nan
                  ).eq(0) * rolling_min_with_offset(df['Low'], 5), 
                  1)
    
    # Задержанные минимумы (без look-ahead)
    delayedlow = (priceMins & (oscMins.rolling(3, min_periods=1).sum().shift(1) > 0)).astype(bool) * df['Low'].shift(1)
    
    # Бычьи дивергенции
    bullish_div1 = ((current_price_bottom < y6) & oscMins & (current_ppo_bottom > prev_ppo_bottom))
    bullish_div2 = ((delayedlow < y6) & (current_ppo_bottom > prev_ppo_bottom))
    
    # Долгосрочные дивергенции
    if long_term_div:
        long_term_bull_filt = valuewhen(priceMins, rolling_min_with_offset(df['Low'], div_lookback_period), 1)
        i4 = (current_ppo_bottom > rolling_min_with_offset(df['PPO'], div_lookback_period))
        i5 = (current_price_bottom < long_term_bull_filt)
        i6 = (delayedlow < long_term_bull_filt)
        
        bullish_div3 = (oscMins & i4 & i5)
        bullish_div4 = (i4 & i6)
    else:
        bullish_div3 = pd.Series(False, index=df.index)
        bullish_div4 = pd.Series(False, index=df.index)
    
    # Комбинируем все дивергенции
    all_bullish = bullish_div1 | bullish_div2 | bullish_div3 | bullish_div4
    
    # Создаем сигналы с продолжительностью signal_duration свечей
    df['bullish_signal'] = 0
    for i in range(len(df)):
        if all_bullish.iloc[i]:
            # Устанавливаем 1 на signal_duration последующих свечей
            end_idx = min(i + signal_duration + 1, len(df))
            df.loc[df.index[i:end_idx], 'bullish_signal'] = 1
    
    # Удаляем промежуточные колонки которые не нужны на выходе
    cols_to_keep = ['PPO', 'bullish_signal']
    result_cols = [col for col in df.columns if col in cols_to_keep]
    
    return df[result_cols + [c for c in df.columns if c not in result_cols]]


# %% [markdown]
# ## TMA_Overlay

# %%
def TMA_Overlay(
    df, 
    show_100_line=True, 
    show_trend_fill=True, 
    show_bullish_3_line=True, 
    show_bullish_engulfing=True,
    eps=1e-10
):
    """
    Ускоренная версия TMA_Overlay с сохранением исходной логики.
    Все скользящие средние нормализованы относительно Close в диапазоне [-1, 1].
    """
    df = df.copy()
    
    # 1. Расчет SMMA через векторизованные операции
    for length in [21, 50, 100, 200]:
        if length == 100 and not show_100_line:
            continue
        
        smma_col = f'smma_{length}'
        # Первое значение SMMA = SMA
        df[smma_col] = df['Close'].rolling(window=length, min_periods=1).mean()
        # Рекурсивный расчет SMMA через формулу
        df[smma_col] = (df[smma_col].shift(1) * (length - 1) + df['Close'])
        df[smma_col] /= length
    
    # 2. Расчет EMA(2)
    df['ema_2'] = df['Close'].ewm(span=2, adjust=False).mean()
    
    # 3. Нормализация скользящих средних
    for col in ['smma_21', 'smma_50', 'smma_100', 'smma_200', 'ema_2']:
        if col in df.columns:
            df[f'{col}_norm'] = (df[col] - df['Close']) / (abs(df['Close']) + eps)
            df.drop(col, axis=1, inplace=True)
    
    # 4. Индикатор тренда (Trend Fill)
    if show_trend_fill:
        df['trend_fill'] = np.where(
            df['ema_2_norm'] > 0, 1, np.where(df['ema_2_norm'] < 0, -1, 0)
        )
    
    # 5. Бычий 3 Line Strike (векторизованная версия)
    if show_bullish_3_line:
        bearish = df['Close'] < df['Open']
        df['bull_3_line'] = 0
        # Условия для 4 свечей подряд
        condition = (
            bearish.shift(3) &  # i-3
            bearish.shift(2) &  # i-2
            bearish.shift(1) &  # i-1
            (df['Close'] > df['Open'].shift(1))  # i > Open(i-1)
        )
        df.loc[condition, 'bull_3_line'] = 1
    
    # 6. Бычий Engulfing (векторизованная версия)
    if show_bullish_engulfing:
        df['bull_engulfing'] = 0
        condition = (
            (df['Open'] <= df['Close'].shift(1)) &  # Open(i) <= Close(i-1)
            (df['Open'] < df['Open'].shift(1)) &   # Open(i) < Open(i-1)
            (df['Close'] > df['Open'].shift(1))     # Close(i) > Open(i-1)
        )
        df.loc[condition, 'bull_engulfing'] = 1
    
    return df
# Применение функции к вашему DataFrame df = TMA_Overlay(df,show_100_line=True,  show_trend_fill=True, show_bullish_3_line=True, show_bullish_engulfing=True)


# %% [markdown]
# ## Объёмный-взвешенный MACD (VW-MACD)

# %%
# Константа для защиты от деления на ноль
EPS = 1e-10

def VW_MACD(df, fast=12, slow=26, signal=9, normalize=True, eps=1e-8):
    """Добавляет VW-MACD (Volume-Weighted MACD) в DataFrame с простой нормализацией."""
    df = df.copy()
    
    # Volume-weighted moving averages
    df['vwma_fast'] = (df['Close'] * df['Volume']).rolling(fast).sum() / (df['Volume'].rolling(fast).sum().replace(0, eps))
    df['vwma_slow'] = (df['Close'] * df['Volume']).rolling(slow).sum() / (df['Volume'].rolling(slow).sum().replace(0, eps))
    
    # MACD components
    df['vw_macd'] = df['vwma_fast'] - df['vwma_slow']
    df['vw_signal'] = df['vw_macd'].ewm(span=signal, adjust=False).mean()
    df['vw_hist'] = df['vw_macd'] - df['vw_signal']
    
    # Удаляем промежуточные колонки
    df.drop(['vwma_fast', 'vwma_slow'], axis=1, inplace=True)

    # Преобразование NaN и бесконечностей
    macd_cols = ['vw_macd', 'vw_signal', 'vw_hist']
    df[macd_cols] = df[macd_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

    # Простая нормализация по глобальному максимуму (по модулю)
    if normalize:
        for col in macd_cols:
            max_val = df[col].abs().max() + eps
            df[col] = df[col] / max_val

    return df


# %% [markdown]
# ## Адаптивный RSI (Adaptive RSI)

# %%
def Adaptive_RSI(df, rsi_period=14, atr_period=14, normalize=True):
    """Добавляет Adaptive RSI в DataFrame."""
    df = df.copy()
    
    # Вычисляем ATR (с заглавными буквами)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean().replace(0, EPS)
    
    # Адаптивный период
    avg_atr = atr.rolling(atr_period).mean().replace(0, EPS)
    volatility_ratio = atr / avg_atr
    adaptive_period = (rsi_period / volatility_ratio).clip(5, 30).fillna(rsi_period).astype(int)
    
    # RSI с переменным окном
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    def calculate_rsi(window):
        avg_gain = window[window > 0].mean()
        avg_loss = -window[window < 0].mean()
        if avg_loss == 0:
            return 100
        rs = avg_gain / (avg_loss + EPS)
        return 100 - (100 / (1 + rs))
    
    df['adaptive_rsi'] = delta.rolling(window=adaptive_period.max()).apply(calculate_rsi, raw=True)
    
    # Нормализация
    if normalize:
        df['adaptive_rsi'] = MinMaxScaler().fit_transform(df[['adaptive_rsi']])
    
    return df


# %% [markdown]
# ## EWO

# %%
def EWO(df, src_col='Close', sma1length=5, sma2length=35, UsePercent=True):
    """
    Добавляет числовой признак Elliott Wave Oscillator для ML моделей
    
    Параметры:
    df - DataFrame с данными
    src_col - колонка с исходными данными (по умолчанию 'Close')
    sma1length - период короткой SMA (по умолчанию 5)
    sma2length - период длинной SMA (по умолчанию 35)
    UsePercent - возвращать разницу в процентах (True) или абсолютное значение (False)
    
    Возвращает:
    DataFrame с добавленной колонкой 'EWO'
    """
    df = df.copy()
    
    # Вычисляем SMA с минимальным периодом 1 (чтобы не было NaN в начале)
    sma1 = df[src_col].rolling(window=sma1length, min_periods=1).mean()
    sma2 = df[src_col].rolling(window=sma2length, min_periods=1).mean()
    
    # Вычисляем EWO (без визуальных меток)
    df['EWO'] = (sma1 - sma2) / df[src_col] * 100 if UsePercent else (sma1 - sma2)
    
    return df

# Для Random Forest (разница в %) 
# df = EWO(df, src_col='Close', UsePercent=True)



# %% [markdown]
# ## fibo_3_lines_dinamic

# %%
def fibo_3_lines_dinamic(df, length=200, src='hlc3', mult=3.0, 
                         lines_to_use=[0.236, 0.5, 1.0], eps=1e-10):
    """
    Возвращает DataFrame с колонками:
    - fibo_upper: 
        *  1.0 — цена на верхней границе (upper_line),
        *  0.0 — цена на средней линии (middle_line),
        * -1.0 — цена сильно выше верхней границы.
    - fibo_basis: нормированное отклонение от средней линии.
    - fibo_lower:
        *  1.0 — цена на нижней границе (lower_line),
        *  0.0 — цена на средней линии,
        * -1.0 — цена сильно ниже нижней границы.
    """
    df = df.copy()
    
    # Вычисляем источник данных (HLC3, Close и т.д.)
    if src == 'hlc3':
        source = (df['High'] + df['Low'] + df['Close']) / 3
    elif src == 'close':
        source = df['Close']
    elif src == 'open':
        source = df['Open']
    elif src == 'high':
        source = df['High']
    elif src == 'low':
        source = df['Low']
    else:
        raise ValueError("Неподдерживаемый источник данных. Используйте 'hlc3', 'close', 'open', 'high' или 'low'")
    
    # Volume-Weighted Moving Average (VWMA)
    cum_vol = source.rolling(window=length).sum()
    cum_vol_price = (source * df['Volume']).rolling(window=length).sum()
    basis = cum_vol_price / (cum_vol + eps)
    
    # Стандартное отклонение и границы
    stdev = source.rolling(window=length).std()
    dev = mult * stdev
    
    upper_line = basis + (lines_to_use[0] * dev)
    middle_line = basis
    lower_line = basis - (lines_to_use[2] * dev)
    
    # 1. fibo_upper: учитываем выход за верхнюю границу
    df['fibo_upper'] = np.select(
        [
            df['Close'] >= upper_line,                     # Цена выше верхней границы → -1
            (df['Close'] > middle_line) & (upper_line > middle_line),  # Между middle и upper → [0, 1]
            df['Close'] <= middle_line                     # Ниже middle → 0
        ],
        [
            -1.0,
            (df['Close'] - middle_line) / (upper_line - middle_line + eps),
            0.0
        ],
        default=0.0
    ).clip(-1, 1)
    
    # 2. fibo_basis: нормированное отклонение от средней линии
    df['fibo_basis'] = (df['Close'] - middle_line) / (dev + eps)
    
    # 3. fibo_lower: учитываем выход за нижнюю границу
    df['fibo_lower'] = np.select(
        [
            df['Close'] <= lower_line,                     # Цена ниже нижней границы → -1
            (df['Close'] < middle_line) & (middle_line > lower_line),  # Между lower и middle → [0, 1]
            df['Close'] >= middle_line                     # Выше middle → 0
        ],
        [
            -1.0,
            (lower_line - df['Close']) / (lower_line - middle_line + eps),
            0.0
        ],
        default=0.0
    ).clip(-1, 1)
    
    return df
# Предположим, ваш DataFrame называется df
# df = fibo_3_lines_dinamic(df)


# %% [markdown]
# ## add_support_resistance_features

# %%
def add_support_resistance_features(df: pd.DataFrame,
                                     window: int = 1000,
                                     lookback: int = 20,
                                     vol_len: int = 2,
                                     tolerance: float = 0.01) -> pd.DataFrame:
    """
    Добавляет признаки поддержки и сопротивления на основе pivot-экстремумов и дельта-объёма,
    рассчитываемые на скользящем окне (например, 1000 свечей).

    Возвращает DF с новыми бинарными колонками:
    - near_support_level
    - near_resistance_level
    - in_support_box
    - in_resistance_box
    """
    df = df.copy()
    eps = 1e-8

    # ATR(200)
    df['ATR'] = atr(high=df['High'], low=df['Low'], close=df['Close'], length=200)

    df['near_support_level'] = 0
    df['near_resistance_level'] = 0
    df['in_support_box'] = 0
    df['in_resistance_box'] = 0

    for i in range(window, len(df)):
        sub = df.iloc[i - window:i]
        close = sub['Close'].values
        open_ = sub['Open'].values
        volume = sub['Volume'].values
        atr_val = df['ATR'].iloc[i]

        # Дельта-объём
        up_vol = np.where(close > open_, volume, 0)
        down_vol = np.where(close < open_, volume, 0)
        delta_vol = up_vol - down_vol

        vol_series = pd.Series(delta_vol)

        vol_hi = vol_series.rolling(vol_len).max().iloc[-1]
        vol_lo = vol_series.rolling(vol_len).min().iloc[-1]

        # Пивоты
        pivots_low = (sub['Low'] == sub['Low'].rolling(lookback, center=True).min())
        pivots_high = (sub['High'] == sub['High'].rolling(lookback, center=True).max())

        support_levels = sub['Low'][pivots_low & (vol_series > vol_hi)].dropna().values
        resistance_levels = sub['High'][pivots_high & (vol_series < vol_lo)].dropna().values

        current_price = df['Close'].iloc[i]

        # Проверка "рядом с уровнем" (в пределах tolerance)
        for lvl in support_levels:
            if abs(current_price - lvl) / (lvl + eps) <= tolerance:
                df.at[df.index[i], 'near_support_level'] = 1
            if lvl <= current_price <= lvl + atr_val:
                df.at[df.index[i], 'in_support_box'] = 1

        for lvl in resistance_levels:
            if abs(current_price - lvl) / (lvl + eps) <= tolerance:
                df.at[df.index[i], 'near_resistance_level'] = 1
            if lvl - atr_val <= current_price <= lvl:
                df.at[df.index[i], 'in_resistance_box'] = 1

    return df.drop(columns=['ATR'])

# %%

# %%
