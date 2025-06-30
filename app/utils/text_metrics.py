import re
from fastapi import logger
import numpy as np
from app.utils.benchmark import benchmark

def calculate_text_metrics(text: str) -> dict:
    """Вычисляет различные метрики текста"""
    if not text:
        return {
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'avg_sentence_length': 0,
            'unique_word_count': 0,
            'lexical_diversity': 0,
            'complexity_score': 0
        }
    
    # Пример реализации (можете заменить на свою)
    words = text.split()
    sentences = [s for s in text.split('.') if s.strip()]
    unique_words = set(words)
    
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    lexical_diversity = len(unique_words) / len(words) if words else 0
    
    # Простая оценка сложности (можете заменить на более сложную)
    complexity_score = min(10, 
                         (avg_word_length * 0.5) + 
                         (avg_sentence_length * 0.3) + 
                         (lexical_diversity * 20))
    
    return {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length,
        'unique_word_count': len(unique_words),
        'lexical_diversity': lexical_diversity,
        'complexity_score': complexity_score
    }

def zero_metrics():
    return {
        'word_count': 0,
        'sentence_count': 0,
        'avg_word_length': 0,
        'avg_sentence_length': 0,
        'unique_word_count': 0,
        'lexical_diversity': 0,
        'complexity_score': 0
    }

def calculate_file_complexity(text: str, file_type: str) -> dict:
    """Calculate file complexity with type adjustments."""
    metrics = calculate_text_metrics(text)  # Используем локальную функцию
    
    type_weights = {
        'pdf': 1.1, 'docx': 1.05, 'pptx': 1.15, 
        'image': 1.2, 'txt': 1.0
    }
    
    weight = type_weights.get(file_type.lower(), 1.0)
    adjusted_score = metrics['complexity_score'] * weight
    
    return {
        'file_type': file_type,
        'basic_metrics': metrics,
        'complexity_score': round(adjusted_score, 2),
        'type_adjustment': round(weight, 2)
    }