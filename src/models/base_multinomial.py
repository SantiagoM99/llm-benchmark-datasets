"""
Base class for single-label (multinomial) classification models.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseMultinomialModel(ABC):
    """Clase base abstracta para modelos de clasificación single-label."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @abstractmethod
    def predict(self, texts: List[str]) -> List[str]:
        """
        Predice etiquetas para una lista de textos.
        
        Args:
            texts: Lista de textos a clasificar
            
        Returns:
            Lista de etiquetas predichas (una por texto)
        """
        pass
    
    @abstractmethod
    def predict_single(self, text: str) -> str:
        """
        Predice etiqueta para un solo texto.
        
        Args:
            text: Texto a clasificar
            
        Returns:
            Etiqueta predicha
        """
        pass