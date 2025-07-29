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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)
from tabulate import tabulate
from datetime import datetime
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import shap
import time
from datetime import timedelta
from sklearn.inspection import permutation_importance
from datetime import timedelta
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.model_selection import cross_val_score
import warnings
from joblib import Parallel, delayed
from sklearn.base import clone
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import numpy as np
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from .all_kline_changes import add_target_column_mod


# %% [markdown]
# **Расчет метрик F1, Precision, recall, порога вхождения (thrashold) и возвращение модели с оптимальными данными**

# %%
def evaluate_model_with_threshold(model, X_train, y_train, X_valid, y_valid, X_test=None, y_test=None):
    """
    Оценивает модель и возвращает результаты в формате для сохранения
    
    Возвращает словарь в формате:
    {
        'model': model,  # обученная модель
        'metrics': {
            'train': {метрики},
            'valid': {метрики},
            'test': {метрики} (если есть),
            'optimal_threshold': float
        },
        'features': list,  # список фичей
        'timestamp': str   # время оценки
    }
    """
    from sklearn.metrics import roc_auc_score
    
    # 1. Получаем предсказанные вероятности
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_valid_proba = model.predict_proba(X_valid)[:, 1]
    
    if X_test is not None and y_test is not None:
        y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # 2. Создаем диапазон порогов
    thresholds = np.linspace(0.01, 0.99, 99)
    
    # 3. Функция для вычисления F1 при разных порогах
    def find_best_threshold(y_true, y_proba, thresholds):
        f1_scores = []
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        best_idx = np.argmax(f1_scores)
        return thresholds[best_idx], f1_scores
    
    # 4. Находим лучшие пороги для train и valid
    train_best_threshold, train_f1_scores = find_best_threshold(y_train, y_train_proba, thresholds)
    valid_best_threshold, valid_f1_scores = find_best_threshold(y_valid, y_valid_proba, thresholds)
    
    # 5. Вычисляем средний оптимальный порог
    optimal_threshold = np.mean([train_best_threshold, valid_best_threshold])
    
    # 6. Создаем словари с метриками
    train_metrics = {
        'thresholds': thresholds,
        'f1_scores': train_f1_scores,
        'precision': [precision_score(y_train, (y_train_proba >= t).astype(int), zero_division=0) for t in thresholds],
        'recall': [recall_score(y_train, (y_train_proba >= t).astype(int), zero_division=0) for t in thresholds],
        'y_proba': y_train_proba,
        'max_f1_threshold': train_best_threshold,
        'roc_auc': roc_auc_score(y_train, y_train_proba)  # Добавлено ROC AUC
    }
    
    valid_metrics = {
        'thresholds': thresholds,
        'f1_scores': valid_f1_scores,
        'precision': [precision_score(y_valid, (y_valid_proba >= t).astype(int), zero_division=0) for t in thresholds],
        'recall': [recall_score(y_valid, (y_valid_proba >= t).astype(int), zero_division=0) for t in thresholds],
        'y_proba': y_valid_proba,
        'max_f1_threshold': valid_best_threshold,
        'roc_auc': roc_auc_score(y_valid, y_valid_proba)  # Добавлено ROC AUC
    }
    
    # 7. Выводим результаты
    print(f"🎯 Лучший порог по F1 (Train): {train_best_threshold:.4f}")
    print(f"🎯 Лучший порог по F1 (Valid): {valid_best_threshold:.4f}")
    print(f"✅ Усредненный оптимальный порог: {optimal_threshold:.4f}")
    print(f"\n📊 ROC AUC Scores:")
    print(f"✅ Train ROC AUC: {train_metrics['roc_auc']:.4f}")
    print(f"✅ Valid ROC AUC: {valid_metrics['roc_auc']:.4f}")
    
    # 8. Считаем финальные метрики с усредненным порогом
    def calculate_final_metrics(y_true, y_proba, threshold, set_name):
        y_pred = (y_proba >= threshold).astype(int)
        metrics = {
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'ROC_AUC': roc_auc_score(y_true, y_proba)  # Добавлено ROC AUC
        }
        print(f"\n📊 {set_name} set (Threshold = {threshold:.4f}):")
        print(f"✅ F1: {metrics['F1']:.4f}")
        print(f"✅ Precision: {metrics['Precision']:.4f}")
        print(f"✅ Recall: {metrics['Recall']:.4f}")
        print(f"✅ ROC AUC: {metrics['ROC_AUC']:.4f}")
        return metrics
    
    train_metrics['final_metrics'] = calculate_final_metrics(y_train, y_train_proba, optimal_threshold, "Train")
    valid_metrics['final_metrics'] = calculate_final_metrics(y_valid, y_valid_proba, optimal_threshold, "Valid")
    
    results = {
        'train': train_metrics,
        'valid': valid_metrics,
        'optimal_threshold': optimal_threshold
    }
    
    if X_test is not None and y_test is not None:
        test_metrics = {
            'thresholds': thresholds,
            'f1_scores': [f1_score(y_test, (y_test_proba >= t).astype(int), zero_division=0) for t in thresholds],
            'precision': [precision_score(y_test, (y_test_proba >= t).astype(int), zero_division=0) for t in thresholds],
            'recall': [recall_score(y_test, (y_test_proba >= t).astype(int), zero_division=0) for t in thresholds],
            'y_proba': y_test_proba,
            'roc_auc': roc_auc_score(y_test, y_test_proba)  # Добавлено ROC AUC
        }
        test_metrics['final_metrics'] = calculate_final_metrics(
            y_test, y_test_proba, optimal_threshold, "Test"
        )
        results['test'] = test_metrics
    
    # 9. Визуализация (остается без изменений)
    plt.figure(figsize=(18, 6))
    
    # 1. Кривые для обучающей выборки
    plt.subplot(1, 3, 1)
    plt.plot(train_metrics['thresholds'], train_metrics['precision'], label='Precision', color='blue')
    plt.plot(train_metrics['thresholds'], train_metrics['recall'], label='Recall', color='green')
    plt.plot(train_metrics['thresholds'], train_metrics['f1_scores'], label='F1', color='red')
    plt.axvline(optimal_threshold, color='k', linestyle='-', label=f'Avg Optimal: {optimal_threshold:.3f}')
    plt.axvline(train_best_threshold, color='b', linestyle=':', label=f'Train Max F1: {train_best_threshold:.3f}')
    plt.title('Train Selection')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    
    # 2. Кривые для валидационной выборки
    plt.subplot(1, 3, 2)
    plt.plot(valid_metrics['thresholds'], valid_metrics['precision'], label='Precision', color='blue')
    plt.plot(valid_metrics['thresholds'], valid_metrics['recall'], label='Recall', color='green')
    plt.plot(valid_metrics['thresholds'], valid_metrics['f1_scores'], label='F1', color='red')
    plt.axvline(optimal_threshold, color='k', linestyle='-', label=f'Avg Optimal: {optimal_threshold:.3f}')
    plt.axvline(valid_best_threshold, color='orange', linestyle=':', label=f'Valid Max F1: {valid_best_threshold:.3f}')
    plt.title('Test Set')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    
    # 3. Сравнение F1-кривых с новым порогом
    plt.subplot(1, 3, 3)
    plt.plot(train_metrics['thresholds'], train_metrics['f1_scores'], label='Train F1', color='blue')
    plt.plot(valid_metrics['thresholds'], valid_metrics['f1_scores'], label='Valid F1', color='orange')
    
    # Добавлена третья линия для тестовой выборки, если она есть
    if X_test is not None and y_test is not None:
        plt.plot(test_metrics['thresholds'], test_metrics['f1_scores'], label='Test F1', color='green')
    
    plt.axvline(optimal_threshold, color='k', linestyle='-', label=f'Avg Optimal: {optimal_threshold:.3f}')
    plt.axvline(train_best_threshold, color='b', linestyle=':', alpha=0.5)
    plt.axvline(valid_best_threshold, color='orange', linestyle=':', alpha=0.5)
    plt.title('F1 Comparison with Optimal Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    
    # 10. Выводим итоговые метрики в таблице (добавляем ROC AUC)
    final_table = [
        ["Dataset", "Threshold Type"] + list(train_metrics['final_metrics'].keys()),
        ["Train", f"Average Optimal ({optimal_threshold:.4f})"] + list(train_metrics['final_metrics'].values()),
        ["Test", f"Average Optimal ({optimal_threshold:.4f})"] + list(valid_metrics['final_metrics'].values())
    ]
    
    if X_test is not None and y_test is not None:
        final_table.append(
            ["Test", f"Average Optimal ({optimal_threshold:.4f})"] + list(results['test']['final_metrics'].values())
        )
    
    print("\nИтоговые метрики со средним оптимальным порогом:")
    print(tabulate(final_table, headers="firstrow", floatfmt=".4f", tablefmt="grid"))

     # Формируем итоговый словарь в нужном формате
    model_package = {
        'model': model,
        'metrics': {
            'train': train_metrics['final_metrics'],
            'valid': valid_metrics['final_metrics'],
            'optimal_threshold': optimal_threshold
        },
        'features': list(X_train.columns),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if X_test is not None and y_test is not None:
        model_package['metrics']['test'] = results['test']['final_metrics']
    
    return model_package


# %%

# %% [markdown]
# Аналогичный расчет с учетом нейросети

# %%
def evaluate_model_with_threshold_mod(model, X_train, y_train, X_valid, y_valid, X_test=None, y_test=None, model_type='sklearn'):
    """
    Оценивает модель (ML или нейросеть) и возвращает результаты с подбором порога
    
    Parameters:
    -----------
    model : object
        Обученная модель (sklearn/lightgbm или keras)
    model_type : str
        Тип модели: 'sklearn' (по умолчанию) или 'keras'
    """
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    # 1. Получаем предсказанные вероятности в зависимости от типа модели
    if model_type == 'keras':
        # Для нейросети
        y_train_proba = model.predict(X_train, verbose=0)
        y_valid_proba = model.predict(X_valid, verbose=0)
        
        if len(y_train_proba.shape) > 1 and y_train_proba.shape[1] > 1:
            y_train_proba = y_train_proba[:, 1] if y_train_proba.shape[1] == 2 else np.argmax(y_train_proba, axis=1)
            y_valid_proba = y_valid_proba[:, 1] if y_valid_proba.shape[1] == 2 else np.argmax(y_valid_proba, axis=1)
            
        if X_test is not None and y_test is not None:
            y_test_proba = model.predict(X_test, verbose=0)
            if len(y_test_proba.shape) > 1 and y_test_proba.shape[1] > 1:
                y_test_proba = y_test_proba[:, 1] if y_test_proba.shape[1] == 2 else np.argmax(y_test_proba, axis=1)
    else:
        # Для классических ML-моделей
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_valid_proba = model.predict_proba(X_valid)[:, 1]
        
        if X_test is not None and y_test is not None:
            y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # 2. Создаем диапазон порогов
    thresholds = np.linspace(0.01, 0.99, 99)
    
    # 3. Улучшенная функция для нахождения баланса между Precision и Recall
    def find_balanced_threshold(y_true, y_proba, thresholds):
        precisions = []
        recalls = []
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
        
        diff = np.abs(np.array(precisions) - np.array(recalls))
        exclude = int(len(thresholds) * 0.1)
        valid_range = range(exclude, len(thresholds)-exclude)
        
        if len(valid_range) > 0:
            best_idx = valid_range[np.argmin(diff[valid_range])]
        else:
            best_idx = np.argmin(diff)
        
        return thresholds[best_idx], precisions, recalls
    
    # 4. Находим сбалансированные пороги
    train_balanced_threshold, train_precisions, train_recalls = find_balanced_threshold(y_train, y_train_proba, thresholds)
    valid_balanced_threshold, valid_precisions, valid_recalls = find_balanced_threshold(y_valid, y_valid_proba, thresholds)
    optimal_threshold = np.mean([train_balanced_threshold, valid_balanced_threshold])
    
    # 5. Функция для расчета финальных метрик
    def calculate_final_metrics(y_true, y_proba, threshold, set_name):
        y_pred = (y_proba >= threshold).astype(int)
        metrics = {
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'ROC_AUC': roc_auc_score(y_true, y_proba)
        }
        print(f"\n📊 {set_name} set (Threshold = {threshold:.4f}):")
        print(f"✅ F1: {metrics['F1']:.4f}")
        print(f"✅ Precision: {metrics['Precision']:.4f}")
        print(f"✅ Recall: {metrics['Recall']:.4f}")
        print(f"✅ ROC AUC: {metrics['ROC_AUC']:.4f}")
        return metrics
    
    # 6. Вычисляем и выводим метрики
    print(f"🎯 Сбалансированный порог (Train): {train_balanced_threshold:.4f}")
    print(f"🎯 Сбалансированный порог (Valid): {valid_balanced_threshold:.4f}")
    print(f"✅ Усредненный оптимальный порог: {optimal_threshold:.4f}")
    
    train_metrics = {
        'thresholds': thresholds,
        'precision': train_precisions,
        'recall': train_recalls,
        'f1_scores': [f1_score(y_train, (y_train_proba >= t).astype(int), zero_division=0) for t in thresholds],
        'y_proba': y_train_proba,
        'balanced_threshold': train_balanced_threshold,
        'final_metrics': calculate_final_metrics(y_train, y_train_proba, optimal_threshold, "Train")
    }
    
    valid_metrics = {
        'thresholds': thresholds,
        'precision': valid_precisions,
        'recall': valid_recalls,
        'f1_scores': [f1_score(y_valid, (y_valid_proba >= t).astype(int), zero_division=0) for t in thresholds],
        'y_proba': y_valid_proba,
        'balanced_threshold': valid_balanced_threshold,
        'final_metrics': calculate_final_metrics(y_valid, y_valid_proba, optimal_threshold, "Valid")
    }
    
    results = {
        'train': train_metrics,
        'valid': valid_metrics,
        'optimal_threshold': optimal_threshold
    }
    
    if X_test is not None and y_test is not None:
        test_balanced_threshold, test_precisions, test_recalls = find_balanced_threshold(y_test, y_test_proba, thresholds)
        test_metrics = {
            'thresholds': thresholds,
            'precision': test_precisions,
            'recall': test_recalls,
            'f1_scores': [f1_score(y_test, (y_test_proba >= t).astype(int), zero_division=0) for t in thresholds],
            'y_proba': y_test_proba,
            'balanced_threshold': test_balanced_threshold,
            'final_metrics': calculate_final_metrics(y_test, y_test_proba, optimal_threshold, "Test")
        }
        results['test'] = test_metrics
    
    # 9. Визуализация с акцентом на область вокруг оптимального порога
    plt.figure(figsize=(18, 6))
    
    # 1. Кривые для обучающей выборки
    plt.subplot(1, 3, 1)
    plt.plot(train_metrics['thresholds'], train_metrics['precision'], label='Precision', color='blue')
    plt.plot(train_metrics['thresholds'], train_metrics['recall'], label='Recall', color='green')
    plt.plot(train_metrics['thresholds'], train_metrics['f1_scores'], label='F1', color='red')
    plt.axvline(optimal_threshold, color='k', linestyle='-', label=f'Avg Optimal: {optimal_threshold:.3f}')
    plt.axvline(train_balanced_threshold, color='b', linestyle=':', label=f'Train Balanced: {train_balanced_threshold:.3f}')
    plt.title('Train Selection')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    
    # 2. Кривые для валидационной выборки
    plt.subplot(1, 3, 2)
    plt.plot(valid_metrics['thresholds'], valid_metrics['precision'], label='Precision', color='blue')
    plt.plot(valid_metrics['thresholds'], valid_metrics['recall'], label='Recall', color='green')
    plt.plot(valid_metrics['thresholds'], valid_metrics['f1_scores'], label='F1', color='red')
    plt.axvline(optimal_threshold, color='k', linestyle='-', label=f'Avg Optimal: {optimal_threshold:.3f}')
    plt.axvline(valid_balanced_threshold, color='orange', linestyle=':', label=f'Valid Balanced: {valid_balanced_threshold:.3f}')
    plt.title('Validation Set')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    
    # 3. Zoom на область вокруг оптимального порога
    plt.subplot(1, 3, 3)
    zoom_range = 0.2  # +/- 20% от оптимального порога
    zoom_min = max(0.01, optimal_threshold - zoom_range)
    zoom_max = min(0.99, optimal_threshold + zoom_range)
    
    mask = (thresholds >= zoom_min) & (thresholds <= zoom_max)
    
    plt.plot(thresholds[mask], np.array(train_metrics['precision'])[mask], label='Train Precision', color='blue', linestyle='--')
    plt.plot(thresholds[mask], np.array(train_metrics['recall'])[mask], label='Train Recall', color='green', linestyle='--')
    plt.plot(thresholds[mask], np.array(valid_metrics['precision'])[mask], label='Valid Precision', color='blue')
    plt.plot(thresholds[mask], np.array(valid_metrics['recall'])[mask], label='Valid Recall', color='green')
    
    plt.axvline(optimal_threshold, color='k', linestyle='-', label=f'Optimal: {optimal_threshold:.3f}')
    plt.title(f'Zoom Around Optimal Threshold ({zoom_min:.2f}-{zoom_max:.2f})')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'model': model,
        'metrics': {
            'train': train_metrics['final_metrics'],
            'valid': valid_metrics['final_metrics'],
            'test': results['test']['final_metrics'] if 'test' in results else None,
            'optimal_threshold': optimal_threshold
        },
        'features': list(X_train.columns) if hasattr(X_train, 'columns') else None,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# %% [markdown]
# **Точечный график распределения целевой переменной**

# %%
def plot_feature_pairplot(df, features_list):
    """
    Строит pairplot для выбранных признаков с выделением целевой переменной 'target'
    
    Параметры:
    ----------
    df : pandas.DataFrame
        Исходный DataFrame с данными
    features_list : list
        Список признаков для визуализации (не включая 'target')
    
    Возвращает:
    -----------
    None (отображает график)
    """
    # Добавляем target к списку признаков
    columns_to_plot = features_list + ['target']
    
    try:
        # Удаляем пропуски и выбираем нужные колонки
        subset_df = df[columns_to_plot].dropna()
        
        # Строим pairplot
        sns.pairplot(subset_df, hue='target', diag_kind='kde', palette='viridis')
        plt.suptitle('Pairplot: selected features vs target', y=1.02)
        plt.show()
        
    except KeyError as e:
        print(f"Ошибка: в DataFrame отсутствует колонка {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

# Пример использования:
# plot_feature_pairplot(df, ['senkou_span_a_norm', 'senkou_span_b_norm', 'breakout_in_5', 'atr_14_norm%', 'EFI'])


# %% [markdown]
# **Корреляционная матрица**

# %%
import matplotlib.pyplot as plt

def plot_correlation_matrix(df, drop_columns=['Data', 'High', 'Low', 'Close', 'Open', 'Volume']):
    """
    Улучшенная версия с лучшей читаемостью для большого числа признаков.
    """
    try:
        # Удаляем ненужные колонки
        data = df.drop(drop_columns, axis=1, errors='ignore')
        
        # Рассчитываем корреляционную матрицу
        corr_matrix = data.corr()
        num_features = len(corr_matrix)
        
        # Динамические настройки в зависимости от количества признаков
        if num_features <= 15:
            figsize = (10, 8)
            font_scale = 1.2
            annot = True
            label_size = 10
        elif num_features <= 30:
            figsize = (16, 14)
            font_scale = 1.0
            annot = False
            label_size = 9
        else:  # Для 50+ признаков
            figsize = (20, 18)
            font_scale = 0.8
            annot = False
            label_size = 8
            # Для очень большого числа признаков можно уменьшить плотность меток
            plt.rcParams['xtick.major.pad'] = 0.5
            plt.rcParams['ytick.major.pad'] = 0.5
        
        # Настройка стиля
        sns.set(font_scale=font_scale)
        plt.figure(figsize=figsize)
        
        # Построение тепловой карты с улучшенными настройками
        heatmap = sns.heatmap(
            corr_matrix,
            cmap='coolwarm',
            annot=annot,
            fmt=".2f",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.7},
            mask=np.triu(np.ones_like(corr_matrix, dtype=bool)),
            annot_kws={"size": 8} if annot else None
        )
        
        # Настройка подписей осей
        heatmap.set_xticklabels(
            heatmap.get_xticklabels(),
            rotation=45,
            ha='right',
            fontsize=label_size
        )
        heatmap.set_yticklabels(
            heatmap.get_yticklabels(),
            rotation=0,
            fontsize=label_size
        )
        
        plt.title(f'Корреляционная матрица ({num_features} признаков)', fontsize=14)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Ошибка при построении графика: {str(e)}")

# Пример использования:
# plot_correlation_matrix(df)


# %% [markdown]
# **Корреляция топ 20**

# %%
def get_top_correlated_pairs(df, top_n=20):
    # Вычисляем матрицу корреляций
    corr_matrix = df.corr().abs()  # Берем модуль корреляции
    
    # Создаем список всех уникальных пар (без дублирования A:B и B:A)
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):  # Исключаем диагональ и зеркальные пары
            pairs.append((cols[i], cols[j], corr_matrix.iloc[i, j]))
    
    # Сортируем пары по убыванию корреляции
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
    
    # Выводим топ-N пар
    print(f"Топ-{top_n} пар по корреляции:")
    for pair in pairs_sorted[:top_n]:
        print(f"{pair[0]} : {pair[1]} : {pair[2]:.4f}")

# Пример использования:
# get_top_correlated_pairs(df, top_n=20)


# %% [markdown]
# **SHAP**

# %%
def explain_model_shap(X_train, model, sample_size=2000, top_n=20, n_jobs = -1):
    """
    Оборачивает расчет SHAP-важности и визуализации признаков
    
    Параметры:
    ----------
    X_train : pd.DataFrame
        Датафрейм признаков
    model : sklearn/xgboost модель
        Обученная модель (RandomForest, LogisticRegression, XGB и др.)
    sample_size : int
        Размер случайной подвыборки
    top_n : int
        Кол-во признаков для отображения
    """
    try:
        total_start_time = time.time()
        model_type = type(model).__name__
        
        print(f"ℹ️ Model type: {model_type}")
        print(f"ℹ️ Number of classes: {getattr(model, 'n_classes_', 'unknown')}")
        
        # 1. Инициализация Explainer
        print("🔄 Initializing SHAP explainer...")
        explainer_start = time.time()
        if model_type in ['RandomForestClassifier', 'RandomForestRegressor', 
                          'XGBClassifier', 'XGBRegressor', 
                          'LGBMClassifier', 'LGBMRegressor']:
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        elif model_type in ['LogisticRegression', 'LinearRegression']:
            explainer = shap.LinearExplainer(model, X_train)
        else:
            explainer = shap.Explainer(model, X_train)
        explainer_time = time.time() - explainer_start
        print(f"✅ SHAP explainer initialized in {timedelta(seconds=explainer_time)}")
        
        # 2. Подвыборка
        sample_size = min(sample_size, len(X_train))
        sample_idx = np.random.choice(X_train.index, size=sample_size, replace=False)
        X_sample = X_train.loc[sample_idx]

        print(f"\n🔄 Calculating SHAP values for {sample_size} samples...")
        shap_start = time.time()

        # Параллельная обработка
        n_jobs = n_jobs
        n_chunks = 4 * (os.cpu_count() or 1)

        def calc_chunk(chunk):
            return explainer.shap_values(chunk, approximate=True, check_additivity=False)

        chunks = np.array_split(X_sample, n_chunks)
        results = Parallel(n_jobs=n_jobs)(delayed(calc_chunk)(chunk) for chunk in chunks)

        # Объединение результатов
        if isinstance(results[0], list):
            shap_values = [np.concatenate([r[i] for r in results]) for i in range(len(results[0]))]
        else:
            shap_values = np.concatenate(results)

        shap_time = time.time() - shap_start
        print(f"✅ SHAP values calculated in {timedelta(seconds=shap_time)}")
        print(f"⏱ Average time per sample: {shap_time/sample_size:.4f} seconds")

        # 3. Обработка SHAP
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else np.mean(shap_values, axis=0)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        print(f"ℹ️ Processed SHAP values shape: {shap_values.shape}")

        # 4. Анализ важности
        print("\n🔄 Calculating feature importance...")
        analysis_start = time.time()
        importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'SHAP_Importance': np.abs(shap_values).mean(axis=0),
            'Direction': np.where(np.mean(shap_values, axis=0) > 0, 'Positive', 'Negative')
        })
        if hasattr(model, 'feature_importances_'):
            importance_df['Model_Importance'] = model.feature_importances_
            importance_df['Model_%'] = 100 * importance_df['Model_Importance'] / importance_df['Model_Importance'].max()

        importance_df['SHAP_%'] = 100 * importance_df['SHAP_Importance'] / importance_df['SHAP_Importance'].max()
        importance_df = importance_df.sort_values('SHAP_%', ascending=False)
        importance_df['Rank'] = range(1, len(importance_df) + 1)
        importance_df['Cumulative_SHAP_%'] = importance_df['SHAP_%'].cumsum()
        analysis_time = time.time() - analysis_start
        print(f"✅ Feature analysis completed in {timedelta(seconds=analysis_time)}")

        # 5. Таблица
        print("\n🔍 Top Features by SHAP Importance:")
        display_cols = ['Rank', 'Feature', 'SHAP_%', 'Direction']
        if 'Model_%' in importance_df.columns:
            display_cols.append('Model_%')
        print(importance_df.head(top_n)[display_cols].to_markdown(index=False, floatfmt=".1f"))

        print("\n📊 Key Metrics:")
        print(f"• Top-5 features explain: {importance_df['Cumulative_SHAP_%'].iloc[4]:.1f}%")
        pos_count = (importance_df['Direction'] == 'Positive').sum()
        neg_count = (importance_df['Direction'] == 'Negative').sum()
        print(f"• Positive/Negative: {pos_count}/{neg_count}")

        # 6. Простая визуализация
        plt.figure(figsize=(10, min(6, top_n * 0.3)))
        colors = importance_df['Direction'].head(top_n).map({'Positive': 'tomato', 'Negative': 'dodgerblue'})
        plt.barh(importance_df['Feature'].head(top_n)[::-1], 
                 importance_df['SHAP_%'].head(top_n)[::-1],
                 color=colors[::-1])
        plt.title(f'Top {top_n} Features by SHAP')
        plt.xlabel('Relative SHAP Importance (%)')
        plt.tight_layout()
        plt.show()

        # 7. Общее время
        total_time = time.time() - total_start_time
        print(f"\n⏱ Total execution time: {timedelta(seconds=total_time)}")
        print("="*50)
        print("Time breakdown:")
        print(f"- Explainer init: {timedelta(seconds=explainer_time)}")
        print(f"- SHAP values: {timedelta(seconds=shap_time)} ({shap_time/total_time*100:.1f}%)")
        print(f"- Analysis: {timedelta(seconds=analysis_time)} ({analysis_time/total_time*100:.1f}%)")

        return importance_df

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if 'shap_values' in locals():
            print(f"SHAP values type: {type(shap_values)}")
            if hasattr(shap_values, 'shape'):
                print(f"SHAP values shape: {shap_values.shape}")
        print(f"X_train shape: {X_train.shape if X_train is not None else 'N/A'}")
        if hasattr(model, 'n_features_in_'):
            print(f"Model features: {model.n_features_in_}")
        return None
# Пример использования:
# explain_model_shap(X_train, logreg_model)
# explain_model_shap(X_train, rf_model, sample_size=2000)


# %% [markdown]
# **Permutation importance (расчет важности признаков при перестановки)**

# %%
def explain_model_permutation(X, y, model, scoring='f1', n_repeats=5, top_n=20, random_state=3):
    """
    Оценивает важность признаков с помощью Permutation Importance.
    
    Параметры:
    ----------
    X : pd.DataFrame
        Признаки (X_train или X_valid)
    y : pd.Series
        Целевая переменная
    model : обученная модель
        RandomForest, LogisticRegression, XGBoost и т.д.
    scoring : str
        Метрика (например, 'f1', 'accuracy', 'roc_auc')
    n_repeats : int
        Количество повторов для случайности
    top_n : int
        Кол-во признаков для отображения
    random_state : int
        Случайное зерно для воспроизводимости
    
    Возвращает:
    -----------
    pd.DataFrame — таблица важности признаков
    """
    try:
        print(f"ℹ️ Model type: {type(model).__name__}")
        print(f"ℹ️ Scoring metric: {scoring}")

        start_time = time.time()

        # Оптимальное количество потоков
        n_jobs = os.cpu_count() - 1 if os.cpu_count() else 1

        print("🔄 Calculating permutation importance...")
        result = permutation_importance(
            model, X, y,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=n_jobs
        )

        elapsed = time.time() - start_time
        print(f"✅ Completed in {timedelta(seconds=elapsed)}")

        # Формируем датафрейм
        importances_df = pd.DataFrame({
            'Feature': X.columns,
            'Mean Importance': result.importances_mean,
            'Std': result.importances_std
        })
        importances_df['Significant'] = importances_df['Mean Importance'] - 2 * importances_df['Std'] > 0
        importances_df = importances_df.sort_values(by='Mean Importance', ascending=False).reset_index(drop=True)
        importances_df['Rank'] = importances_df.index + 1

        print("\n🔍 Top Features by Permutation Importance:")
        display_cols = ['Rank', 'Feature', 'Mean Importance', 'Std', 'Significant']
        print(importances_df.head(top_n)[display_cols].to_markdown(index=False, floatfmt=".3f"))

        # Простая визуализация
        top_features = importances_df.head(top_n)
        plt.figure(figsize=(10, min(6, top_n * 0.3)))
        bars = plt.barh(top_features['Feature'][::-1], top_features['Mean Importance'][::-1],
                        xerr=top_features['Std'][::-1], color='mediumseagreen')
        plt.xlabel("Mean Importance")
        plt.title(f"Top {top_n} Features by Permutation Importance")
        plt.tight_layout()
        plt.show()

        return importances_df

    except Exception as e:
        print(f"❌ Error during permutation importance: {e}")
        return None


# %% [markdown]
# **RFECV**

# %%
def show_rfecv_results(X_train, y_train, estimator, 
                      scoring='f1', step=1, n_splits=3, 
                      n_jobs=-1, verbose=0):
    """
    Выполняет отбор признаков с помощью RFECV и показывает результаты:
    - список отобранных признаков
    - график зависимости качества от количества признаков
    
    Параметры:
    ----------
    X_train : pd.DataFrame или array-like
        Матрица признаков для обучения
    y_train : pd.Series или array-like
        Вектор целевой переменной
    estimator : объект модели
        Уже инициализированная модель (например, RandomForestClassifier())
    scoring : str, default='f1'
        Метрика для оценки
    step : int, default=1
        Количество удаляемых признаков на каждой итерации
    n_splits : int, default=3
        Количество фолдов для TimeSeriesSplit
    n_jobs : int, default=-1
        Количество ядер для параллельных вычислений
    verbose : int, default=0
        Уровень детализации вывода
    """
    
    # TimeSeries split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # RFECV с переданной моделью и TSCV
    rfecv_selector = RFECV(
        estimator=estimator,
        step=step,
        cv=tscv,
        scoring=scoring,
        verbose=verbose,
        n_jobs=n_jobs
    )
    
    # Копируем данные, чтобы избежать предупреждений
    X_rfecv = X_train.copy() if isinstance(X_train, pd.DataFrame) else X_train
    y_rfecv = y_train.copy()
    
    # Выполняем отбор признаков
    rfecv_selector.fit(X_rfecv, y_rfecv)
    
    # Получаем имена отобранных признаков
    if isinstance(X_train, pd.DataFrame):
        rfecv_features = X_train.columns[rfecv_selector.support_].tolist()
    else:
        rfecv_features = [f"feature_{i}" for i, selected in enumerate(rfecv_selector.support_) if selected]
    
    # Выводим результаты
    print("\n✅ Признаки, отобранные RFECV:")
    print(rfecv_features)
    
    # Строим график
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(rfecv_selector.cv_results_['mean_test_score']) + 1),
             rfecv_selector.cv_results_['mean_test_score'])
    plt.xlabel("Количество признаков")
    plt.ylabel(f"{scoring} score")
    plt.title(f"Качество модели ({estimator.__class__.__name__}) в зависимости от числа признаков (RFECV)")
    plt.grid(True)
    plt.show()


# %% [markdown]
# **Boruta**

# %%
def explain_model_boruta(
    X_train, 
    X_valid, 
    y_train, 
    y_valid, 
    model, 
    max_iter=100, 
    n_splits=3, 
    random_state=3,
    perc=100,
    alpha=0.05,
    two_step=True,
    n_estimators='auto',
    verbose=0
):
    """
    Выполняет отбор признаков с помощью Boruta и TimeSeriesSplit.

    Параметры:
    ----------
    X_train : pd.DataFrame
        Признаки обучающей выборки
    X_valid : pd.DataFrame
        Признаки валидационной выборки (не используется, включен для единообразия)
    y_train : pd.Series
        Целевая переменная обучающей выборки
    y_valid : pd.Series
        Целевая переменная валидационной выборки (не используется)
    model : объект
        Обучаемая модель (RandomForest, XGBoost, LogisticRegression и др.)
    max_iter : int, default=100
        Максимальное количество итераций Boruta (чем больше, тем тщательнее отбор,
        но дольше выполнение)
    n_splits : int, default=3
        Количество разбиений в TimeSeriesSplit (для временных рядов)
    random_state : int, default=3
        Зерно генератора случайных чисел для воспроизводимости
    perc : int, default=100
        Процент отбора признаков для сравнения с шумом (меньшие значения делают
        отбор более консервативным)
    alpha : float, default=0.05
        Уровень значимости для отбраковки гипотез (меньше = строже отбор)
    two_step : bool, default=True
        Использовать двухэтапную процедуру (True рекомендуется для больших наборов данных)
    n_estimators : int или 'auto', default='auto'
        Количество деревьев в ансамбле (если 'auto', берется из модели)
    verbose : int, default=0
        Уровень детализации вывода (0 - нет вывода, 1 - базовый, 2 - подробный)
    """
    try:
        print(f"ℹ️ Model type: {type(model).__name__}")
        print(f"ℹ️ Boruta params: max_iter={max_iter}, perc={perc}, alpha={alpha}")
        print(f"ℹ️ TimeSeriesSplit n_splits: {n_splits}")

        X_df = X_train.copy()
        X_array, y_array = X_df.values, y_train.values

        tscv = TimeSeriesSplit(n_splits=n_splits)

        boruta = BorutaPy(
            estimator=model,
            n_estimators=n_estimators,
            verbose=verbose,
            random_state=random_state,
            max_iter=max_iter,
            perc=perc,
            alpha=alpha,
            two_step=two_step
        )

        print("🔄 Running Boruta feature selection on time series splits...")

        support_masks = []
        for i, (train_idx, test_idx) in enumerate(tscv.split(X_array)):
            print(f"  • Fold {i+1}/{n_splits} — Train size: {len(train_idx)}, Test size: {len(test_idx)}")
            X_fold_train, y_fold_train = X_array[train_idx], y_array[train_idx]
            boruta.fit(X_fold_train, y_fold_train)
            support_masks.append(boruta.support_.copy())

        # Пересечение масок по всем фолдам
        final_support = np.all(support_masks, axis=0)
        selected_features = X_df.columns[final_support].tolist()

        print(f"\n✅ Итоговые отобранные признаки ({len(selected_features)}):")
        for i, feat in enumerate(selected_features, 1):
            print(f"{i:>2}. {feat}")

        return selected_features

    except Exception as e:
        print(f"❌ Ошибка при выполнении Boruta: {str(e)}")
        return None


# %% [markdown]
# **Mutual Information (взаимная информация)**

# %%
def explain_model_mutual_info(X_train, y_train, top_n=20, random_state=3):
    """
    Расчёт важности признаков на основе Mutual Information.

    Параметры:
    ----------
    X_train : pd.DataFrame
        Обучающая выборка признаков
    y_train : pd.Series
        Целевая переменная
    top_n : int
        Количество признаков для отображения
    random_state : int
        Зерно генератора случайных чисел для MI-оценки
    """
    try:
        start_time = time.time()
        print(f"ℹ️ Calculating Mutual Information for {X_train.shape[1]} features...")

        # 1. Расчёт MI
        mi_scores = mutual_info_classif(X_train, y_train, random_state=random_state)
        mi_df = pd.DataFrame({
            'Feature': X_train.columns,
            'MI_Score': mi_scores
        }).sort_values('MI_Score', ascending=False)

        elapsed_time = time.time() - start_time
        print(f"✅ MI calculation completed in {elapsed_time:.2f} seconds")

        # 2. Таблица топ-N
        print(f"\n🔍 Top {top_n} Features by Mutual Information:")
        print(mi_df.head(top_n).to_markdown(index=False, floatfmt=".4f"))

        # 3. Визуализация
        plt.figure(figsize=(10, min(6, top_n * 0.3)))
        plt.barh(mi_df['Feature'].head(top_n)[::-1], 
                 mi_df['MI_Score'].head(top_n)[::-1], 
                 color='skyblue')
        plt.xlabel('Mutual Information Score')
        plt.title(f'Top {top_n} Features by Mutual Information')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Ошибка при расчёте Mutual Information: {str(e)}")


# %% [markdown]
# **Granger Causality (причинность по Грейнджеру)**

# %%
def explain_model_granger(X_train, y_train, target_name='target', max_lag=5, top_n=20):
    """
    Анализ Granger Causality между признаками и целевой переменной.

    Параметры:
    ----------
    X_train : pd.DataFrame
        Признаки обучающей выборки
    y_train : pd.Series
        Целевая переменная
    target_name : str
        Название целевой переменной (для подписи)
    max_lag : int
        Максимальное количество лагов для теста Грейнджера
    top_n : int
        Количество признаков для отображения
    """
    try:
        start_time = time.time()
        print(f"ℹ️ Calculating Granger causality for {X_train.shape[1]} features...")

        def check_granger_causality(feature_series, target_series, max_lag=5):
            data = pd.DataFrame({
                'target': target_series,
                'feature': feature_series
            }).dropna()
            try:
                test_result = grangercausalitytests(data[['target', 'feature']], maxlag=max_lag, verbose=False)
                p_values = [test_result[i+1][0]['ssr_chi2test'][1] for i in range(max_lag)]
                return np.min(p_values)
            except:
                return np.nan  # В случае ошибки вернуть NaN

        granger_results = {
            feature: check_granger_causality(X_train[feature], y_train)
            for feature in X_train.columns
        }

        granger_df = pd.DataFrame({
            'Feature': list(granger_results.keys()),
            'Granger_p_value': list(granger_results.values())
        }).dropna().sort_values('Granger_p_value')

        elapsed_time = time.time() - start_time
        print(f"✅ Granger analysis completed in {elapsed_time:.2f} seconds")

        # Вывод таблицы
        print(f"\n🔍 Top {top_n} Features by Granger Causality (lowest p-values):")
        print(granger_df.head(top_n).to_markdown(index=False, floatfmt=".4e"))

        # Визуализация
        plt.figure(figsize=(10, min(6, top_n * 0.3)))
        plt.barh(granger_df['Feature'].head(top_n)[::-1],
                 -np.log10(granger_df['Granger_p_value'].head(top_n)[::-1]),
                 color='salmon')
        plt.xlabel(r'$-\log_{10}$(p-value)')
        plt.title(f'Top {top_n} Features by Granger Causality vs "{target_name}"')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Ошибка при анализе Granger Causality: {str(e)}")


# %% [markdown]
# **Зависимость F1 модели от количества деревьев**

# %%
def evaluate_n(n, base_model, cv_splits, X_train_arr, y_train_arr):
    """Глобальная функция для оценки модели с заданным n_estimators."""
    model = clone(base_model).set_params(
        n_estimators=n,
        warm_start=True,  # Инкрементальное обучение
        verbose=0
    )
    scores = []
    for train_idx, test_idx in cv_splits:
        X_train_fold = X_train_arr[train_idx]
        X_test_fold = X_train_arr[test_idx]
        y_train_fold = y_train_arr[train_idx]
        y_test_fold = y_train_arr[test_idx]
        
        model.fit(X_train_fold, y_train_fold)
        pred = model.predict(X_test_fold)
        scores.append(f1_score(y_test_fold, pred))
    return np.mean(scores)

def plot_f1_vs_n_estimators(X_train, y_train, base_model, 
                          n_estimators_range=(100, 1000), step=50, 
                          n_splits=3, n_jobs=-1):
    """
    Оптимизированная версия с использованием joblib.Parallel.
    """
    # Проверка входных данных
    if not hasattr(base_model, 'fit') or not hasattr(base_model, 'predict'):
        raise ValueError("base_model должен быть моделью scikit-learn с методами fit и predict")
    
    n_estimators_values = np.arange(
        n_estimators_range[0], 
        n_estimators_range[1] + 1, 
        step
    )
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_splits = list(tscv.split(X_train))
    
    # Конвертация в numpy array для ускорения
    X_train_arr = X_train.values if hasattr(X_train, 'iloc') else X_train
    y_train_arr = y_train.values if hasattr(y_train, 'iloc') else y_train
    
    # Игнорирование предупреждений
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        print(f"⏳ Анализ {len(n_estimators_values)} значений n_estimators...")
        
        # Параллельные вычисления через joblib
        f1_scores = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(evaluate_n)(
                n, base_model, cv_splits, X_train_arr, y_train_arr
            ) for n in tqdm(n_estimators_values, desc="n_estimators")
        )
        
        # Фильтрация NaN результатов
        valid_mask = ~np.isnan(f1_scores)
        n_estimators_values = n_estimators_values[valid_mask]
        f1_scores = np.array(f1_scores)[valid_mask]
        
        if not len(f1_scores):
            print("⚠️ Все вычисления завершились с ошибкой!")
            return
        
        # Построение графика
        plt.figure(figsize=(12, 6))
        plt.plot(n_estimators_values, f1_scores, 'b-o', alpha=0.7)
        plt.title(f'Зависимость F1 от n_estimators ({base_model.__class__.__name__})')
        plt.xlabel('n_estimators')
        plt.ylabel('F1-score (CV среднее)')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        best_idx = np.argmax(f1_scores)
        plt.scatter(
            n_estimators_values[best_idx], f1_scores[best_idx],
            color='red', s=150, zorder=5,
            label=f'Лучшее: {f1_scores[best_idx]:.4f} (n={n_estimators_values[best_idx]})'
        )
        
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        print(f"✅ Оптимальное количество деревьев: {n_estimators_values[best_idx]} с F1={f1_scores[best_idx]:.4f}")


# %% [markdown]
# **Распределение целевой переменной внутри выборок**

# %%
def show_class_balance(y, y_train, y_valid, y_test):
    # Собираем данные в таблицу
    balance_df = pd.DataFrame({
        'Весь датасет': y.value_counts(normalize=True).round(3),
        'Обучающая': y_train.value_counts(normalize=True).round(3),
        'Валидационная': y_valid.value_counts(normalize=True).round(3),
        'Тестовая': y_test.value_counts(normalize=True).round(3)
    }).fillna(0)  # на случай отсутствующих классов
    
    # Выводим таблицу в стиле "plain"
    print("📊 Баланс классов (доли):")
    print(
        balance_df.to_markdown(
            tablefmt="simple",  # Чистый формат без лишних линий
            stralign="center",  # Выравнивание по центру
            floatfmt=".3f"       # Формат чисел
        )
    )
    
    # Визуализация
    plt.figure(figsize=(10, 5))
    balance_df.plot(kind='bar', width=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.title('Распределение классов по выборкам', pad=20)
    plt.ylim(0, 1)
    plt.ylabel('Доля класса')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(framealpha=0.9)
    plt.tight_layout()
    plt.show()


# %% [markdown]
# **Подбор оптимального соотношения риск прибыль**

# %%
def evaluate_parameters(df, X, train_index, valid_index, 
                       target_candles=20,
    
                       rr_thresholds=np.arange(1.5, 4.1, 0.5), 
                       targets=np.arange(0.001, 0.0061, 0.0005)):
    
    results = []
    
    # Перебор всех комбинаций параметров
    for rr in tqdm(rr_thresholds, desc='Processing rr_threshold'):
        for target in tqdm(targets, desc='Processing target', leave=False):
            # Создание целевой переменной с текущими параметрами
            df_temp = df.copy()
            df_temp = add_target_column_mod(
                df_temp,
                target_candles=target_candles,
                target=target,
                rr_threshold=rr
            )
            
            # Берем только целевую переменную из модифицированного датафрейма
            y = df_temp['target'].values
            
            # Разделение данных с использованием предопределенных индексов
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
            
            # Пропускаем итерацию если в valid нет обоих классов
            if len(np.unique(y_valid)) < 2:
                continue
                
            # Обучение модели
            model = RandomForestClassifier(
            n_estimators=400,
            max_depth=13,
            max_features=0.8,
            min_samples_leaf=6,
            min_samples_split=5,
            max_samples=0.7,
            min_impurity_decrease=0.0002,
            class_weight={0: 1, 1: 5},
            criterion='entropy',
            bootstrap=True,
            random_state=3,
            n_jobs=-1)
            # model = lgb.LGBMClassifier(
            # num_leaves=30,
            # max_depth=9,
            # learning_rate=0.05,
            # n_estimators=200,
            # min_child_samples=50,
            # subsample=0.5,
            # colsample_bytree=0.8,
            # reg_alpha=0.1,
            # reg_lambda=0.1,
            # scale_pos_weight=1,
            # boosting_type='gbdt',
            # importance_type='split',
            # random_state=3,
            # verbose=-1)
            
            model.fit(X_train, y_train)
            
            # Прогноз и расчет метрик
            y_pred = model.predict(X_valid)
            results.append({
                'rr_threshold': rr,
                'target': target,
                'f1': f1_score(y_valid, y_pred),
                'precision': precision_score(y_valid, y_pred),
                'recall': recall_score(y_valid, y_pred),
                'positive_ratio': np.mean(y_train)
            })
    
    return pd.DataFrame(results)

# %%
