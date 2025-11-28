"""
Predictor for single-label (multinomial) classification tasks.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import weave

from models.base_multinomial import BaseMultinomialModel
from utils.multinomial_datareader import MultinomialDataset


class MultinomialPredictor:
    """Genera predicciones y las guarda en formato JSON o parquet."""
    
    def __init__(self, model: BaseMultinomialModel, dataset: MultinomialDataset):
        self.model = model
        self.dataset = dataset
    
    @weave.op()
    def predict_split(
        self, 
        split: str, 
        max_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Genera predicciones para un split del dataset.
        
        Args:
            split: 'train', 'dev', o 'test'
            max_samples: Límite de muestras (para testing)
            
        Returns:
            DataFrame con textos, labels reales, y labels predichas
        """
        texts, true_labels = self.dataset.get_texts_and_labels(split)
        
        if max_samples:
            texts = texts[:max_samples]
            true_labels = true_labels[:max_samples]
        
        predicted_labels = self.model.predict(texts)
        
        results = pd.DataFrame({
            "input_text": texts,
            "expected": true_labels,
            "prediction": predicted_labels
        })
        
        return results
    
    def save_predictions(
        self, 
        split: str, 
        output_dir: str,
        max_samples: Optional[int] = None,
        format: str = "json",
        shot_type: str = "zero_shot"
    ) -> Path:
        results_df = self.predict_split(split, max_samples)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_clean = self.model.model_name.replace("/", "_")
        filename = f"{model_name_clean}_{shot_type}_{split}_{timestamp}"
        
        if format == "json":
            filepath = output_path / f"{filename}.json"
            output_data = {
                "model": self.model.model_name,
                "split": split,
                "shot_type": shot_type,
                "timestamp": timestamp,
                "num_samples": len(results_df),
                "results": results_df.to_dict(orient="records")
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
        else:
            filepath = output_path / f"{filename}.parquet"
            results_df.to_parquet(filepath, index=False)
        
        print(f"Predicciones guardadas en: {filepath}")
        print(f"Total de muestras: {len(results_df)}")
        
        return filepath