"""
Base class for Named Entity Recognition (NER) models.
"""
from abc import ABC, abstractmethod
from typing import List, Dict


class BaseNERModel(ABC):
    """Clase base abstracta para modelos NER."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @abstractmethod
    def predict(self, sentences: List[str]) -> List[List[Dict[str, str]]]:
        """
        Extrae entidades de una lista de oraciones.
        
        Args:
            sentences: Lista de oraciones (strings)
            
        Returns:
            Lista de listas de entidades [{"text": ..., "type": ...}, ...]
        """
        pass
    
    @abstractmethod
    def predict_single(self, sentence: str) -> List[Dict[str, str]]:
        """
        Extrae entidades de una sola oración.
        
        Args:
            sentence: Oración a procesar
            
        Returns:
            Lista de entidades [{"text": ..., "type": ...}, ...]
        """
        pass