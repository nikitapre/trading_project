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
#     display_name: Python [conda env:base] *
#     language: python
#     name: conda-base-py
# ---

# %%
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
from datetime import datetime
import time
import math
import requests
import random
from .all_kline_changes import apply_indicators, add_target_column_mod, apply_main_indicators  # Импорт из соседнего файла


# %% [markdown]
# ### Получение свечей с Bybit и обработка

# %%
# Получение свечей и обработка

def fetch_kline_data(symbol, tf, start_ms, end_ms, 
                    max_retries=5, 
                    max_consecutive_failures=4,
                    batch_size=200):
    """
    Улучшенная версия функции для загрузки свечных данных с Bybit
    с обработкой ошибки 403 и улучшенным логированием
    """
    url = "https://api.bybit.com/v5/market/kline"
    ms_tf = int(tf) * 60 * 1000
    batch_count = math.ceil((end_ms - start_ms) / (batch_size * ms_tf))
    
    all_data = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://www.bybit.com',
        'Accept': 'application/json'
    }
    
    consecutive_failures = 0
    failed_batches = 0

    for i in range(batch_count):
        batch_start = start_ms + i * batch_size * ms_tf
        batch_end = min(end_ms, batch_start + batch_size * ms_tf)
        
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': str(tf),
            'start': int(batch_start),
            'end': int(batch_end),
            'limit': str(batch_size)
        }

        current_batch_failed = True
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Добавляем случайную задержку между запросами
                time.sleep(random.uniform(0.1, 0.5))
                
                r = requests.get(url, params=params, headers=headers, timeout=15)
                
                # Специальная обработка 403 ошибки
                if r.status_code == 403:
                    error_msg = r.json().get('retMsg', 'Unknown error')
                    raise requests.exceptions.HTTPError(
                        f"403 Forbidden: {error_msg}. Проверьте: "
                        "1) Корректность символа\n"
                        "2) Доступность рынка\n"
                        "3) Ограничения API ключей\n"
                        "4) IP-ограничения"
                    )
                
                r.raise_for_status()
                data = r.json()
                
                # Улучшенная проверка структуры ответа
                if not data.get('result') or not isinstance(data['result'], dict):
                    raise ValueError("Некорректная структура ответа API")
                
                result_list = data['result'].get('list', [])
                if not result_list:
                    print(f"⚠️ {symbol} | Батч {i+1}/{batch_count} | Пустой массив данных")
                    break
                
                # Используем вашу существующую функцию обработки
                df = process_batch_data(result_list)
                all_data.append(df)
                consecutive_failures = 0
                current_batch_failed = False
                break
                
            except requests.exceptions.HTTPError as e:
                last_error = str(e)
                if '403' in last_error:
                    print(f"🔒 Критическая ошибка доступа: {last_error}")
                    return None
                print(f"❌ HTTP Error [{symbol} | Батч {i+1}/{batch_count} | Попытка {attempt+1}/{max_retries}]: {last_error[:200]}")
                time.sleep(2 ** attempt + random.uniform(0.1, 1.0))
            except Exception as e:
                last_error = str(e)
                print(f"❌ Ошибка [{symbol} | Батч {i+1}/{batch_count} | Попытка {attempt+1}/{max_retries}]: {last_error[:200]}")
                time.sleep(2 ** attempt + random.uniform(0.1, 1.0))
        
        if current_batch_failed:
            consecutive_failures += 1
            failed_batches += 1
            print(f"🚫 Батч {i+1}/{batch_count} не загружен. Последняя ошибка: {last_error[:200]}")
            
            if consecutive_failures >= max_consecutive_failures:
                print(f"⏩ Прерываем загрузку {symbol} (подряд {consecutive_failures} неудачных батчей)")
                break

    # Формирование результата
    if not all_data:
        print(f"❌ Не удалось загрузить данные для {symbol} после {failed_batches} неудачных батчей")
        return None
        
    final_df = pd.concat(all_data).drop_duplicates().sort_values('Data').reset_index(drop=True)
    
    if failed_batches > 0:
        print(f"🟡 {symbol} загружен частично: {len(all_data)}/{batch_count} батчей")
    
    return final_df

def process_batch_data(result):
    """Обработка успешно загруженного батча"""
    data = pd.DataFrame(result)
    return pd.DataFrame({
        'Data': pd.to_datetime(data.iloc[:, 0].astype('float'), unit='ms'),
        'Open': data.iloc[:, 1].astype('float'),
        'High': data.iloc[:, 2].astype('float'),
        'Low': data.iloc[:, 3].astype('float'),
        'Close': data.iloc[:, 4].astype('float'),
        'Volume': data.iloc[:, 5].astype('float')
    }).sort_values('Data')

def kline_candles(symbol, tf, start=None, end=None, n_candles=None, 
                 length=20, target_candles=5, target=0.05, rr_threshold=2,
                 min_completeness=0.7):  # Новый параметр: минимальная полнота данных (0.7 = 70%)
    try:
        ms_tf = int(tf) * 60 * 1000

        # Подготовка временного диапазона
        if start and end:
            start_dt = datetime.strptime(start, "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(end, "%Y-%m-%d %H:%M")
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)
            expected_candles = ((end_ms - start_ms) / ms_tf) + 1
        elif n_candles:
            end_ms = int(time.time() * 1000)
            start_ms = end_ms - n_candles * ms_tf
            expected_candles = n_candles
        else:
            return pd.DataFrame()

        # Загрузка данных
        full_df = fetch_kline_data(symbol, tf, start_ms, end_ms)
        
        # Проверка результата
        if full_df is None:
            print(f"❌ Не удалось загрузить данные для {symbol}")
            return pd.DataFrame()
            
        if full_df.empty:
            print(f"⚠️ Пустой DataFrame для {symbol}")
            return pd.DataFrame()

        # Проверка полноты данных
        actual_candles = len(full_df)
        completeness = actual_candles / expected_candles
        
        if completeness < min_completeness:
            print(f"⚠️ {symbol} загружен не полностью: {actual_candles}/{int(expected_candles)} свечей ({completeness:.1%})")
            return pd.DataFrame()

        # Применение индикаторов
        #full_df = apply_main_indicators(full_df, length)
        full_df = add_target_column_mod(full_df, target_candles, target, rr_threshold)
        full_df.dropna(inplace=True)
        
        if full_df.empty:
            print(f"⚠️ Нет данных после обработки для {symbol}")
            
        return full_df
        
    except Exception as e:
        print(f"⚠️ Ошибка в kline_candles для {symbol}: {str(e)[:200]}")
        return pd.DataFrame()


# %% [markdown]
# ### Получение данных с пк

# %%
def load_kline_data_from_csv(file_path, length=20, target_candles=50, target=0.025, rr_threshold =2):
    start_all = time.time()

    # Загрузка
    
    df = pd.read_csv(file_path)
    

    expected_columns = {'Data', 'Open', 'High', 'Low', 'Close', 'Volume'}
    if not expected_columns.issubset(df.columns):
        raise ValueError(f"CSV должен содержать столбцы: {expected_columns}")

    
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values('Data').reset_index(drop=True)
    

    # Индикаторы
    t0 = time.time()
    #apply_indicators(full_df, length) возвращает все 55 индикаторов
    #apply_main_indicators(full_df, length)  возвращает выбранные лучшие индикаторы
    df_ind = apply_main_indicators(df.copy(), length)
    print(f"📈 Индикаторы: {time.time() - t0:.2f} сек")

    # Объединение
    t0 = time.time()
    new_columns = [col for col in df_ind.columns if col not in df.columns]
    df = pd.concat([df, df_ind[new_columns]], axis=1)
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    print(f"🔗 Объединение: {time.time() - t0:.2f} сек")

    # Целевые переменные
    
    df = add_target_column_mod(df, target_candles, target, rr_threshold)
    

    # Очистка NaN
    rows_before = df.shape[0]
    cols_before = df.shape[1]
    total_nans_before = df.isna().sum().sum()

    df.dropna(inplace=True)

    rows_after = df.shape[0]
    cols_after = df.shape[1]
    total_nans_after = df.isna().sum().sum()
    

    print(f"🧼 Удалено строк с NaN: {rows_before - rows_after}")
    print(f"🧼 Количество NaN до: {total_nans_before}, после: {total_nans_after}")
    print(f"📊 Размер таблицы: до — {rows_before}x{cols_before}, после — {rows_after}x{cols_after}")

    print(f"⏱️ Общая длительность: {time.time() - start_all:.2f} сек")
    return df

# %% [markdown]
#

# %%

# %%

# %%
