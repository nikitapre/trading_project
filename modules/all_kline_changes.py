# ---
# jupyter:
#   jupytext:
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

# %%
import numpy as np
import datetime
import time
import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.stats import linregress
from scipy.fft import rfft, rfftfreq
from sklearn.linear_model import Ridge
from scipy.stats import wasserstein_distance  # Альтернатива, если нет POT
import ot
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
from .indicators import (add_breakout_flags,
    get_fibonacci_flags,
    calc_stochastic_rsi,
    find_bullish_rsi_divergence,
    add_normalized_volume_profile,
    add_volume_filter,
    add_atr_normalized, add_normalized_ichimoku, add_normalized_ichimoku_on_rsi, add_normalized_ichimoku_htf,
    compute_atr_trailing,
    compute_trix_elder,
    compute_combined_indicators,
    different_MA,
    VWAP,
    add_liquidity_imbalance, add_hidden_divergence,
    add_price_acceleration,
    quarter_theory, add_all_features, kagi_conversion_line, calculate_LISS,add_breakdown_flags,
    hurst_exponent, sample_entropy, dominant_frequency, nar_residual, wasserstein_distance, Keltner_func,
    compute_cmf, compute_rsi, super_trend, fibo_dinamic, add_structural_features, add_macd, add_volume_features, add_candle_vol_size_features,
    different_EMA, calculate_rsi_slope, calculate_atr_slope, calculate_atr_ema_normalized, add_choppiness_index, add_bollinger_features, bollinger_awesome_alert,
    add_parabolic_sar_feature, add_wave_phase_position, add_fft_phase_position, adaptive_zscore, quantile_position, atr_position, rsi_stochastic_hybrid,
    add_close_window_norm_pca, add_close_window_columns, add_atr_24_norm_dynamics, add_volume_dynamics, rsi_divergence, ppo, TMA_Overlay, EWO, fibo_3_lines_dinamic, VW_MACD, Adaptive_RSI,
    add_atr_features, add_atr_regimes, add_price_atr_interaction, atr_local_extremes, add_atr_normalized_rsi_weight,
    different_EMA_binary, VWAP_binary, volume_base_indicators, volume_base_indicators_binary, bullish_volume_dominance_binary, add_rsi_level_signal,
    add_support_resistance_features, fib_dist_050_last50, add_vwap_features_with_norm, calculate_resistance_distance, calculate_support_distance
    
)


# %% [markdown]
# ### Модифицированный y

# %%
def add_target_column_mod(
    df,
    target_candles=20,
    target=0.04,
    rr_threshold=2.0
):
    """
    Присваивает y=1 всем свечам, где в следующие N свечей:
    - цена достигает TP раньше, чем SL (или SL не достигается вообще).
    Если TP и SL достигаются на одной свече — SL считается приоритетным (y=0).
    """
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    y = np.zeros(len(df), dtype=int)

    sl_pct = target / rr_threshold  # SL = target / rr_threshold

    for i in range(len(df)):
        entry_price = close[i]
        tp_price = entry_price * (1 + target)
        sl_price = entry_price * (1 - sl_pct)

        window_end = min(i + target_candles + 1, len(df))
        tp_hit_first = False

        for j in range(i + 1, window_end):
            hit_sl = low[j] <= sl_price
            hit_tp = high[j] >= tp_price

            if hit_sl and hit_tp:
                # SL и TP на одной свече → SL считается первым → y=0
                break
            elif hit_sl:
                # SL раньше → y=0
                break
            elif hit_tp:
                # TP раньше → y=1
                tp_hit_first = True
                break

        if tp_hit_first:
            y[i] = 1

    df['target'] = y
    return df
#Пример как вызвать
# df = add_target_column_no_overlap(
#         df,
#         target_pct=0.025,
#         target_candles=20,
#         rr_threshold=2.0
#     )


# %% [markdown]
# ### Добавление основных индикаторов

# %%
def apply_main_indicators(df, length=20, eps=1e-8):
    if df is None or df.empty:
        raise ValueError("Input DataFrame is None or empty")
    
    try:
        
        df['RSI21'] = ta.rsi(df['Close'], length=21) / 100
        df['RSI_50_21_diff'] = (ta.rsi(df['Close'], length=50) / 100) - df['RSI21']
        
        deviations = different_EMA(df)
        # Объединяем с исходным DataFrame (по индексу)
        df = pd.concat([df, deviations], axis=1)
        df = add_vwap_features_with_norm(df)
        df = VWAP(df)
        df['dump_return_15'] = ta.ema(df['Close'], 9).pct_change(periods=15) / 100
        df['fib_dist_050_last50'] = fib_dist_050_last50(df)
        
        return df
    except Exception as e:
        print(f"Error in apply_main_indicators: {str(e)}")
        return None  # Или return df для частичных результатов


# %% [markdown]
# ### Добавочные Индикаторы

# %%
def apply_add_indicators(df, length=20, eps=1e-8):
    if df is None or df.empty:
        raise ValueError("Input DataFrame is None or empty")

    try:
        print(f'Начальное количество NaN в df: {df.isna().sum().sum()}')
        t0 = time.time()
        df = add_atr_normalized(df, windows=[24], normalization_window=200)
        print(f"⏱️ resistance_slope_dist_200: {time.time() - t0:.2f} сек")

        
       
        print(f'Итоговое количество NaN в df: {df.isna().sum().sum()}')
        return df

    except Exception as e:
        print(f"Error in apply_add_indicators: {str(e)}")
        return df  # Лучше вернуть df, пусть даже частично заполненный


# %% [markdown]
# ### Добавление индикаторов

# %%
def apply_indicators(df, length):
    
    eps = 1e-8
    timings = {}

    # Ценовой lag + z-score
    t0 = time.time()
    # ======Лаг цены======        
    window = 100  # Размер скользящего окна для статистик
    lags = [1, 3, 8]  # Лаги для разностных признаков
    log_lags = [1, 5, 8]  # Лаги для логарифмических признаков
        
    # 1. Разностные признаки с rolling-статистиками c window/5
    for lag in lags:
        df[f'Close_lag{lag}_diff_ratio'] = (
            (df['Close'].shift(lag) - df['Close'].shift(lag + 1)) / 
            (df['Close'].shift(lag + 1).rolling(window // 5).mean() + eps)
        )            
    # 2. Логарифмические признаки с защитой от нуля
    for lag in log_lags:
        df[f'Close_log_lag{lag}'] = np.log(
            np.abs(df['Close'].shift(lag)) / 
            (df['Close'].shift(lag).rolling(window).mean() + eps)
        )
    # =========================================================
    timings['lag + z-score'] = time.time() - t0

    #Ускорение изм цены
    df = add_price_acceleration(df)


    # Линейная регрессия
    t0 = time.time()
    # Вычисление линейной регрессии с помощью pandas_ta
    df['linreg'] = ta.linreg(close=df['Close'], length=length) / (df['Close'] + eps) - 1
    df['linreg_lag'] = df['linreg'].shift(length)
    timings['linreg'] = time.time() - t0

    # Объем: производные
    t0 = time.time()
    df = add_volume_features(df, window=20)
    avg_volume = df['Volume'].rolling(20).mean()
    df['accel_volume'] = df['Volume'].diff(5).diff() / (avg_volume + eps)
    df['accel_volume_fast'] = df['Volume'].diff().diff() / (avg_volume + eps)
    timings['accel_volume'] = time.time() - t0

    # Пробой
    t0 = time.time()
    df = add_breakout_flags(df, resistance_window=50, lookback=int(length / 2))
    timings['breakout'] = time.time() - t0

    # Фибоначчи
    # t0 = time.time()
    # fib_flags = get_fibonacci_flags(df, window=300, threshold=0.05)
    # df = pd.concat([df, fib_flags], axis=1)
    # timings['fibonacci'] = time.time() - t0

    # RSI
    t0 = time.time()
    df['RSI7'] = ta.rsi(df['Close'], length=7)
    df['RSI21'] = ta.rsi(df['Close'], length=21)
    df['RSI100'] = ta.rsi(df['Close'], length=100)
    df['RSI21_diff'] = df['RSI21'].diff(int(length/2))
    timings['rsi'] = time.time() - t0
    print(f"⏱️ rsi: {timings['rsi']:.2f} сек")

    # Объёмные индикаторы
    t0 = time.time()
    obv_diff = ta.obv(close=df['Close'], volume=df['Volume']).diff()
    df['OBV'] = obv_diff / (df['Volume'] + eps)

    ad_diff = ta.volume.ad(df['High'], df['Low'], df['Close'], df['Volume']).diff()
    df['AD'] = ad_diff / (df['Volume'] + eps)

    efi = ta.volume.efi(close=df['Close'], volume=df['Volume'])
    df['EFI'] = efi / (df['Volume'].rolling(10).mean() + eps)

    df['CMF'] = ta.cmf(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], length=length)
    df['MFI'] = ta.mfi(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], length=length)
    timings['volume_indicators'] = time.time() - t0

    # Stochastic RSI
    t0 = time.time()
    df = calc_stochastic_rsi(df, rsi_period=14, stoch_period=14, smooth_k=3, smooth_d=5)
    timings['stoch_rsi'] = time.time() - t0


    # Профиль объема
    t0 = time.time()
    df = add_normalized_volume_profile(df, window=100, lag=3, change_window=5, sma_window=10)
    timings['volume_profile'] = time.time() - t0

    # Объёмный фильтр
    t0 = time.time()
    df = add_volume_filter(df, use_ema=True)
    timings['volume_filter'] = time.time() - t0

    # ATR
    t0 = time.time()
    df = add_atr_normalized(df, windows=[7, 24, 72])
    timings['atr'] = time.time() - t0

    # Ишимоку
    t0 = time.time()
    df = add_normalized_ichimoku(df)
    timings['ichimoku'] = time.time() - t0

    # Ишимоку htf
    t0 = time.time()
    df = add_normalized_ichimoku_htf(df)
    timings['ichimoku'] = time.time() - t0
    print(f"⏱️ ichimoku: {timings['ichimoku']:.2f} сек")

    # Ишимоку RSI
    t0 = time.time()
    df = add_normalized_ichimoku_on_rsi(df, rsi_period=9, prefix="rsi9_ichimoku")
    df = add_normalized_ichimoku_on_rsi(df, rsi_period=14, prefix="rsi14_ichimoku")
    df = add_normalized_ichimoku_on_rsi(df, rsi_period=50, prefix="rsi50_ichimoku")
    print(f"⏱️ Ишимоку RSI: {time.time() - t0:.2f} сек")

    # TRIX + Elder Ray
    t0 = time.time()
    trix_elder = compute_trix_elder(df)
    trix_elder_new_cols = [col for col in trix_elder.columns if col not in df.columns]
    df = df.join(trix_elder[trix_elder_new_cols])
    timings['trix + elder'] = time.time() - t0

    # STC + Fisher + Keltner
    t0 = time.time()
    new_features = compute_combined_indicators(df)
    df[new_features.columns] = new_features
    timings['stc + fisher + keltner'] = time.time() - t0

    #different MA
    t0 = time.time()
    # Вычисляем отклонения
    deviations = different_MA(df)
    # Объединяем с исходным DataFrame (по индексу)
    df = pd.concat([df, deviations], axis=1)

    deviations = different_EMA(df)
    # Объединяем с исходным DataFrame (по индексу)
    df = pd.concat([df, deviations], axis=1)
    timings['different MA'] = time.time() - t0

    #vwap
    t0 = time.time()
    df = VWAP(df)    
    timings['VWAP'] = time.time() - t0

    #liquidity_imbalance
    t0 = time.time()
    df = add_liquidity_imbalance(df)
    timings['liquidity_imbalance'] = time.time() - t0


    #Liquidity Imbalance Short-Squeeze Score" (LISS)
    t0 = time.time()
    df['LISS'] = calculate_LISS(df)
    timings['LISS'] = time.time() - t0
    print(f"⏱️ LISS: {timings['LISS']:.2f} сек")


    # Новые фичи
    t0 = time.time()
    df = add_all_features(df)
    print(f"⏱️ Новые фичи: {time.time() - t0:.2f} сек")

    # Теория четвертей
    t0 = time.time()
    df = quarter_theory(df)
    print(f"⏱️ Теория четвертей: {time.time() - t0:.2f} сек")

    # Пробой short
    t0 = time.time()
    df = add_breakdown_flags(df, support_window=50, lookback=int(length / 2))
    print(f"⏱️ Пробой short: {time.time() - t0:.2f} сек")

    # Расчет ML индикаторов
    t0 = time.time()
    df['wasserstein_100'] = df['Close'].rolling(100).apply(lambda x: wasserstein_distance(x, 100, 50), raw=True)        
    df['wasserstein_20'] = df['Close'].rolling(20).apply(lambda x: wasserstein_distance(x, 20, 10), raw=True)       
    df['hurst'] = df['Close'].rolling(100).apply(lambda x: hurst_exponent(x, max_lag=100, poly_deg=1), raw=True)            
    df['entropy'] = df['Close'].rolling(50).apply(sample_entropy, raw=True)              
    df['fft_freq'] = df['Close'].rolling(64).apply(dominant_frequency, raw=True)                
    df['nar_res'] = df['Close'].rolling(20).apply(nar_residual, raw=True)
    print(f"⏱️ Расчет ML индикаторов: {time.time() - t0:.2f} сек")   
    
    
    #Keltner_func
    df = Keltner_func(df)

    # --- RSI и производные признаки
    rsi = compute_rsi(df['Close'], period=14, eps=eps)
    df['RSI_14_manual'] = rsi
    df['RSI_14_zscore'] = ((rsi - rsi.mean()) / (rsi.std(ddof=0) + eps)).fillna(0)

    df['RSI_14_ma'] = rsi.rolling(20).mean()
    df['RSI_dist_from_mean'] = rsi - df['RSI_14_ma']

    # --- CMF и дивергенции
    cmf = compute_cmf(df, length=length, eps=eps)
    df['cmf_manual'] = cmf

    df['cmf_price_div'] = ((cmf.diff() > 0) & (df['Close'].diff() < 0)).astype(int)
    df['rsi_price_div'] = ((rsi.diff() > 0) & (df['Close'].diff() < 0)).astype(int)

    # Super trend
    t0 = time.time()
    df = super_trend(df, period=50, multiplier=1.3)  
    df = super_trend(df, period=50, multiplier=5.0)   
    df = super_trend(df, period=7, multiplier=1.5)
    df = super_trend(df, period=7, multiplier=5.0)
    print(f"⏱️ Расчет Super trend: {time.time() - t0:.2f} сек")   

    # fibo_dinamic
    t0 = time.time()
    df = fibo_dinamic(df, period=300)
    print(f"⏱️ Расчет fibo_dinamic: {time.time() - t0:.2f} сек")   
    
    # Отчет
    print("⏱️ Время по блокам:")
    for name, duration in timings.items():
        print(f" - {name}: {duration:.2f} сек")

    return df


# %% [markdown]
# ### Удаление ненужных признаков

# %%
def drop_low_importance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет заранее определённые признаки с низкой важностью, если они присутствуют в датафрейме.
    Основано на анализе importance mean ≤ 0 и importance std ≤ 0.001, а также высокой корреляции.

    Parameters:
    - df (pd.DataFrame): входной датафрейм.

    Returns:
    - pd.DataFrame: датафрейм без указанных признаков.
    """
    features_to_drop = [
        # Фичи с нулевой/отрицательной важностью и низкой стабильностью
        'lower_low_count', 'AD', 'body_signed', 'bull_inside_bar', 'MFI',
        'breakout_now', 'bull_hammer_manual', 'impulse_candle_detected',
        'order_block_volume', 'bull_morning_star_manual', 'bull_pin_bar',
        'OBV', 'volume_ratio', 'bull_three_white_manual', 'dist_to_fib_500',
        'stoch_cross_bull', 'accel_volume', 'accel_volume_fast',
        'higher_high_count', 'VP_Change_Norm', 'weighted_direction', 'rsi_low_5', 'divergence_power_20',
        'has_divergence_20', 'divergence_power_5', 'has_divergence_5', 'rsi_low_20', 'price_low_20', 'weighted_supertrend',
        'price_low_5', 'Close_lag1_z', 'dynamic_direction', 'dynamic_supertrend', 'supertrend_direction',
        
        # Дополнительные фичи с очень низкой важностью (mean < 0.005)
        'fib_touch_count_786', 'CMF', 'stoch_rsi_k', 'sma10',
        'bos', 'dist_to_fib_382', 'fib_touch_count_618', 'hybrid_supertrend',
        'hybrid_direction', 'supertrend',
        
        # Удаление корелирующих признаков
        'sma20', 'dist_to_fib_618', 'VP_SMA_Norm',
        
        # Новые фичи для удаления на основе permutation importance и корреляций
        'EFI', 'stoch_rsi_d', 'norm_adx',  # Отрицательная/нулевая важность
        'RSI7_lag', 'RSI21_lag',    # Сильно коррелируют с другими RSI
        'VP_Norm_Lag',            # Высокая корреляция между собой
        'optimized_direction',               # Почти идентичен rsi_trend_combo
        'ADX_20',                            # Нулевая важность
        'atr_14',
            'STC',                   # Нулевая Permutation Importance
    
    'senkou_span_a_norm',    # Отрицательная важность
    'Fisher'                # Отрицательная важность
    
    ]
    #'volume_filter','fib_touch_count_236', 'VP_Norm'
     
    # Удаляем только те фичи, которые есть в датафрейме
    return df.drop(columns=[col for col in features_to_drop if col in df.columns], errors='ignore')


# %% [markdown]
# ### Оставить только нужные фичи

# %%
def keep_selected_features(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только выбранные фичи в DataFrame, проверя их наличие и удаляя дубликаты."""
    top_features = ['target',       
    'Data',     
    'Volume', 'Open', 'Close', 'High', 'Low',

    'senkou_span_a_norm',
     'senkou_span_b_norm',
     'KeltnerWidth',
     'atr_14_norm',
     'breakout_in_5',
     'rsi50_ichimoku_senkou_a_norm',
     'RSI7_1H_norm',
     'kijun_sen_norm',
     'dist_to_fib_786',
     'tenkan_sen_norm_htf',
     'linreg',
     'TRIX',
     'vwma20',
     'volume_filter' ]
    
    # Удаление дубликатов в исходном DF (на случай если они есть)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    # Проверка отсутствующих колонок
    missing_features = [f for f in top_features if f not in df.columns]
    # if missing_features:
    #     print(f"Предупреждение: В DataFrame отсутствуют {len(missing_features)} важных колонок:") #ЗАКОММЕНТИРОВАЛ ВЫВОД ОШИБОК
    #     print(missing_features)
    
    # Фильтрация: оставляем только существующие колонки из top_features
    existing_features = [f for f in top_features if f in df.columns]
    
    # Удаление колонок, не входящих в топ (если они есть в DF)
    extra_features = [f for f in df.columns if f not in top_features]
    # if extra_features:
    #     print(f"Удалено {len(extra_features)} лишних колонок, не входящих в топ-фичи") #ЗАКОММЕНТИРОВАЛ ВЫВОД ОШИБОК
    
    return df[existing_features].copy()
#ЗАКОММЕНТИРОВАЛ ВЫВОД ОШИБОК
# Пример использования:
# df_processed = keep_selected_features(raw_df)

# %%

# %%

# %%
