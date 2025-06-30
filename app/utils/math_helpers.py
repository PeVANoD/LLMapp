import numpy as np
from typing import List

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Вычисление косинусной схожести между векторами"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def moving_average(values: List[float], window_size: int) -> List[float]:
    """Скользящее среднее для временных рядов"""
    return np.convolve(values, np.ones(window_size)/window_size, mode='valid')