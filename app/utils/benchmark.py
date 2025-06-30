import numpy as np
import time
from collections import deque
import logging

logger = logging.getLogger(__name__)

class FileProcessingBenchmark:
    def __init__(self, window_size=100):
        self.stats = {
            file_type: {
                'times': deque(maxlen=window_size),
                'complexities': deque(maxlen=window_size),
                'ratios': deque(maxlen=window_size)
            } 
            for file_type in ['pdf', 'docx', 'pptx', 'image', 'text']
        }
    
    def add_metrics(self, file_type, processing_time, complexity):
        if file_type not in self.stats:
            logger.warning(f"Unknown file type: {file_type}")
            return
        
        # Исправлено: добавлена проверка на нулевое время обработки
        if processing_time <= 0:
            logger.warning(f"Invalid processing time: {processing_time} for {file_type}")
            return
            
        # Исправлено: защита от нулевой сложности
        if complexity <= 0:
            complexity = 0.1
            
        ratio = processing_time / complexity
        
        self.stats[file_type]['times'].append(processing_time)
        self.stats[file_type]['complexities'].append(complexity)
        self.stats[file_type]['ratios'].append(ratio)
    
    def get_advanced_stats(self, file_type):
        if file_type not in self.stats or not self.stats[file_type]['times']:
            return None
            
        data = self.stats[file_type]
        times = list(data['times'])
        complexities = list(data['complexities'])
        
        # Исправлено: проверка минимального количества точек
        if len(times) < 2:
            return {
                'count': len(times),
                'time_stats': {'mean': np.mean(times) if times else 0},
                'complexity_stats': {'mean': np.mean(complexities) if complexities else 0},
                'correlation': 0,
                'error': "Insufficient data points for correlation"
            }
        
        # Расчет корреляции с обработкой ошибок
        try:
            correlation = np.corrcoef(times, complexities)[0, 1]
        except Exception as e:
            logger.error(f"Correlation error for {file_type}: {str(e)}")
            correlation = 0
        
        return {
            'count': len(times),
            'time_stats': {
                'mean': np.mean(times),
                'percentile_90': np.percentile(times, 90) if times else 0
            },
            'complexity_stats': {
                'mean': np.mean(complexities),
                'max': np.max(complexities) if complexities else 0
            },
            'correlation': correlation
        }
        
    def get_all_stats(self):
        return {ft: self.get_advanced_stats(ft) for ft in self.stats.keys()}

    def time_it(self, file_type):
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                self.add_metrics(file_type, end_time - start_time, 1.0)
                return result
            return wrapper
        return decorator

# Глобальный экземпляр для бенчмаркинга
benchmark = FileProcessingBenchmark()