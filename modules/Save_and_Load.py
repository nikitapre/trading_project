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
import joblib
import os
from typing import Dict, Any

# %% [markdown]
# **Load model**

# %%
import os
import joblib
from typing import Dict, Any

def load_model_with_threshold(model_name: str) -> Dict[str, Any]:
    """
    Улучшенная версия функции. Автоматически ищет model_exports:
    - Сначала проверяет папку рядом с модулем (../model_exports)
    - Если нет — пытается найти в текущей директории (старый вариант)
    """
    # Путь к model_exports относительно расположения модуля
    module_dir = os.path.dirname(os.path.abspath(__file__))
    model_exports_relative = os.path.join(module_dir, "..", "model_exports")
    
    # Варианты путей для поиска
    possible_paths = [
        model_exports_relative,  # ../model_exports (новый вариант)
        "model_exports"          # Текущая директория (старый вариант)
    ]
    
    # Ищем существующую папку model_exports
    for base_path in possible_paths:
        filename = f"{model_name}_with_threshold.pkl"
        filepath = os.path.join(base_path, filename)
        
        if os.path.exists(filepath):
            loaded = joblib.load(filepath)
            
            # Проверка структуры данных
            required_keys = {'model', 'params', 'threshold'}
            if not all(key in loaded for key in required_keys):
                missing = required_keys - set(loaded.keys())
                raise ValueError(
                    f"В файле {filepath} отсутствуют ключи: {missing}"
                )
            
            print(f"✅ Модель '{model_name}' загружена из {filepath}")
            return loaded
    
    # Если ни один путь не сработал
    available_models = []
    for base_path in possible_paths:
        if os.path.exists(base_path):
            available_models.extend([
                f.replace("_with_threshold.pkl", "")
                for f in os.listdir(base_path)
                if f.endswith('.pkl')
            ])
    
    raise FileNotFoundError(
        f"Файл модели '{model_name}' не найден. Доступные модели:\n"
        f"{chr(10).join(sorted(set(available_models)))}"  # Исправлено: добавлена скобка
    )


# %% [markdown]
# **Save model**

# %%
import os
import joblib
from typing import Any, Dict, List
def save_model_with_threshold(
    model_name: str,
    model: Any,
    params: Dict[str, Any],
    threshold: float,
    features: List[str],  # Теперь List определен
    compress: int = 3
) -> None:
    """
    Сохраняет модель, параметры, порог и названия признаков в папку model_exports с возможностью сжатия.
    
    Параметры:
    ----------
    model_name : str
        Название модели (без расширения), например 'logreg_model'.
    model : Any
        Обученная модель.
    params : dict
        Параметры модели.
    threshold : float
        Порог классификации.
    features : List[str]
        Список названий признаков.
    compress : int, optional (default=3)
        Уровень сжатия (0-9), где 0 - без сжатия, 9 - максимальное сжатие.
    """
    # Создаем папку, если ее нет
    os.makedirs('model_exports', exist_ok=True)
    
    # Формируем имя файла
    filename = f"{model_name}_with_threshold.pkl"
    filepath = os.path.join('model_exports', filename)
    
    # Подготавливаем данные для сохранения
    to_save = {
        'model': model,
        'params': params,
        'threshold': threshold,
        'features': features
    }
    
    # Сохраняем с указанным уровнем сжатия
    joblib.dump(to_save, filepath, compress=compress)
    print(f"✅ Модель сохранена в {filepath} (сжатие: уровень {compress})")

# %%
