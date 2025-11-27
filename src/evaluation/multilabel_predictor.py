import pandas as pd
from pathlib import Path
from typing import Optional
from utils.multilabel_datareader import MultiLabelDataset
from models.base_multilabel import BaseMultiLabelModel


class MultiLabelPredictor:
    """Genera predicciones y las guarda en formato parquet."""
    
    def __init__(self, model: BaseMultiLabelModel, dataset: MultiLabelDataset):
        self.model = model
        self.dataset = dataset
        
    def predict_split(self, split: str, batch_size: Optional[int] = None) -> pd.DataFrame:
        """
        Genera predicciones para un split del dataset.
        
        Args:
            split: 'train', 'dev', o 'test'
            batch_size: Tamaño de batch para procesar (None = todo de una vez)
            
        Returns:
            DataFrame con textos, labels reales, y labels predichas
        """
        texts, true_labels = self.dataset.get_texts_and_labels(split)
        
        if batch_size:
            predicted_labels = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                predictions = self.model.predict(batch)
                predicted_labels.extend(predictions)
        else:
            predicted_labels = self.model.predict(texts)
        
        # Crear DataFrame con resultados
        results = pd.DataFrame({
            'input_text': texts,
            'true_labels': true_labels,
            'predicted_labels': predicted_labels
        })
        
        return results
    
    def save_predictions(self, split: str, output_path: str, 
                        batch_size: Optional[int] = None) -> None:
        """
        Genera predicciones y las guarda en parquet.
        
        Args:
            split: 'train', 'dev', o 'test'
            output_path: Path donde guardar el archivo
            batch_size: Tamaño de batch para procesar
        """
        results = self.predict_split(split, batch_size)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        results.to_parquet(output_file, index=False)
        print(f"Predicciones guardadas en: {output_file}")
        print(f"Total de muestras: {len(results)}")