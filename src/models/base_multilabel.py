# src/models/base_model.py
from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseMultiLabelModel(ABC):
    """Clase base abstracta para modelos de clasificación multilabel."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @abstractmethod
    def predict(self, texts: List[str]) -> List[List[str]]:
        """
        Predice etiquetas para una lista de textos.
        
        Args:
            texts: Lista de textos a clasificar
            
        Returns:
            Lista de listas de etiquetas predichas
        """
        pass
    
    @abstractmethod
    def predict_with_scores(self, texts: List[str]) -> List[Tuple[List[str], List[float]]]:
        """
        Predice etiquetas con scores de confianza.
        
        Args:
            texts: Lista de textos a clasificar
            
        Returns:
            Lista de tuplas (labels, scores) para cada texto
        """
        pass