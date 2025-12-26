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
        
    def predict_split(
        self, 
        split: str, 
        batch_size: Optional[int] = None,
        max_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Genera predicciones para un split del dataset.
        
        Args:
            split: 'train', 'dev', o 'test'
            batch_size: Tamaño de batch para procesar (None = todo de una vez)
            max_samples: Número máximo de muestras a procesar (None = todas)
            
        Returns:
            DataFrame con textos, labels reales, y labels predichas
        """
        texts, true_labels = self.dataset.get_texts_and_labels(split)
        
        # Limit number of samples if specified
        if max_samples is not None and max_samples > 0:
            original_size = len(texts)
            texts = texts[:max_samples]
            true_labels = true_labels[:max_samples]
            print(f"Processing {len(texts)} samples (limited from {original_size})")
        
        if batch_size:
            predicted_labels = []
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            for i in range(0, len(texts), batch_size):
                batch_num = i // batch_size + 1
                batch = texts[i:i + batch_size]
                print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} samples)...")
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

    def load_few_shot_examples(self, split: str = "train", max_examples: int = 10):
        """
        Load few-shot examples from a dataset split.
        
        Args:
            split: Dataset split to load examples from (usually 'train')
            max_examples: Maximum number of examples to load
        """
        try:
            # Intentar obtener textos y labels del split
            texts, labels_list = self.dataset.get_texts_and_labels(split)
        except (KeyError, ValueError, AttributeError) as e:
            print(f"Warning: Could not load split '{split}' for few-shot examples: {e}")
            return
        
        # Limitar el número de ejemplos
        n_examples = min(max_examples, len(texts))
        
        if n_examples == 0:
            print(f"Warning: No examples found in split '{split}'")
            return
        
        examples = []
        for i in range(n_examples):
            text = texts[i]
            labels = labels_list[i]
            
            # Truncate text if too long
            if len(text) > 200:
                text = text[:200] + "..."
            
            examples.append((text, labels))
        
        # Set examples in the prompt template
        self.model.prompt_template.set_examples(examples)
        print(f"✓ Loaded {len(examples)} few-shot examples from '{split}' split")
    
    def save_predictions(
        self, 
        split: str, 
        output_path: str, 
        batch_size: Optional[int] = None,
        max_samples: Optional[int] = None
    ) -> None:
        """
        Genera predicciones y las guarda en parquet.
        
        Args:
            split: 'train', 'dev', o 'test'
            output_path: Path donde guardar el archivo
            batch_size: Tamaño de batch para procesar
            max_samples: Número máximo de muestras a procesar
        """
        results = self.predict_split(split, batch_size, max_samples)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        results.to_parquet(output_file, index=False)
        print(f"Predicciones guardadas en: {output_file}")
        print(f"Total de muestras: {len(results)}")