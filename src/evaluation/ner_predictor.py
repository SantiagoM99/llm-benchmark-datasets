"""
Predictor for NER tasks.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import weave

from models.base_ner import BaseNERModel
from utils.ner_datareader import NERDataset


class NERPredictor:
    """Genera predicciones NER y las guarda en formato JSON."""
    
    def __init__(self, model: BaseNERModel, dataset: NERDataset):
        self.model = model
        self.dataset = dataset
    
    @weave.op()
    def predict_split(
        self, 
        split: str, 
        max_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        Genera predicciones para un split del dataset.
        
        Args:
            split: 'train', 'dev', o 'test'
            max_samples: Límite de muestras (para testing)
            
        Returns:
            Lista de resultados con texto, entidades esperadas y predichas
        """
        sentences = self.dataset.get_text_sentences(split)
        tokens_list, tags_list = self.dataset.get_sentences_and_labels(split)
        
        if max_samples:
            sentences = sentences[:max_samples]
            tokens_list = tokens_list[:max_samples]
            tags_list = tags_list[:max_samples]
        
        # Convertir tags BIO a entidades
        expected_entities = [
            self._bio_to_entities(tokens, tags) 
            for tokens, tags in zip(tokens_list, tags_list)
        ]
        
        # Predecir entidades
        predicted_entities = self.model.predict(sentences)
        
        results = []
        for i, (sent, exp, pred) in enumerate(zip(sentences, expected_entities, predicted_entities)):
            results.append({
                "sentence": sent,
                "expected_entities": exp,
                "predicted_entities": pred
            })
        
        return results
    
    def _bio_to_entities(
        self, 
        tokens: List[str], 
        tags: List[str]
    ) -> List[Dict[str, str]]:
        """
        Convierte tags BIO a lista de entidades.
        
        Args:
            tokens: Lista de tokens
            tags: Lista de tags BIO
            
        Returns:
            Lista de entidades [{"text": ..., "type": ...}, ...]
        """
        entities = []
        current_entity = None
        current_tokens = []
        
        for token, tag in zip(tokens, tags):
            if tag.startswith("B-"):
                # Guardar entidad anterior si existe
                if current_entity:
                    entities.append({
                        "text": " ".join(current_tokens),
                        "type": current_entity
                    })
                # Iniciar nueva entidad
                current_entity = tag[2:]  # Quitar "B-"
                current_tokens = [token]
            elif tag.startswith("I-") and current_entity:
                # Continuar entidad actual
                current_tokens.append(token)
            else:
                # Tag "O" - guardar entidad anterior si existe
                if current_entity:
                    entities.append({
                        "text": " ".join(current_tokens),
                        "type": current_entity
                    })
                    current_entity = None
                    current_tokens = []
        
        # Guardar última entidad si existe
        if current_entity:
            entities.append({
                "text": " ".join(current_tokens),
                "type": current_entity
            })
        
        return entities
    
    def save_predictions(
        self, 
        split: str, 
        output_dir: str,
        max_samples: Optional[int] = None,
        shot_type: str = "zero_shot"
    ) -> Path:
        """
        Guarda predicciones en formato JSON.
        """
        results = self.predict_split(split, max_samples)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_clean = self.model.model_name.replace("/", "_")
        filename = f"{model_name_clean}_{shot_type}_{split}_{timestamp}.json"
        
        filepath = output_path / filename
        output_data = {
            "model": self.model.model_name,
            "split": split,
            "shot_type": shot_type,
            "timestamp": timestamp,
            "num_samples": len(results),
            "entity_types": self.dataset.entity_types,
            "results": results
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"Predicciones guardadas en: {filepath}")
        print(f"Total de muestras: {len(results)}")
        
        return filepath