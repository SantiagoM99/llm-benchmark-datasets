import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np



class MultiLabelDataset:
    """
    Clase para cargar y gestionar datasets multilabel en formato parquet.
    
    Attributes:
        data_dir: Directorio que contiene los archivos parquet
        train: DataFrame de entrenamiento
        dev: DataFrame de desarrollo/validación
        test: DataFrame de prueba
        labels: Lista de etiquetas únicas en el dataset
    """
    
    def __init__(self, data_dir: str):
        """
        Inicializa el dataset cargando los archivos parquet.
        
        Args:
            data_dir: Path al directorio que contiene train.parquet, dev.parquet, test.parquet
        """
        self.data_dir = Path(data_dir)
        self._validate_files()
        
        self.train = pd.read_parquet(self.data_dir / "train.parquet")
        self.dev = pd.read_parquet(self.data_dir / "dev.parquet")
        self.test = pd.read_parquet(self.data_dir / "test.parquet")
        
        self.labels = self._extract_labels()
        
    def _validate_files(self) -> None:
        """Valida que existan los tres archivos requeridos."""
        required_files = ["train.parquet", "dev.parquet", "test.parquet"]
        for file in required_files:
            if not (self.data_dir / file).exists():
                raise FileNotFoundError(f"No se encontró {file} en {self.data_dir}")
    
    def _extract_labels(self) -> List[str]:
        """Extrae todas las etiquetas únicas del dataset."""
        all_labels = set()
        for df in [self.train, self.dev, self.test]:
            for label_array in df["labels"]:
                all_labels.update(label_array)
        return sorted(list(all_labels))
    
    def get_split(self, split: str) -> pd.DataFrame:
        """
        Obtiene un split específico del dataset.
        
        Args:
            split: 'train', 'dev', o 'test'
            
        Returns:
            DataFrame correspondiente al split solicitado
        """
        if split == "train":
            return self.train
        elif split == "dev":
            return self.dev
        elif split == "test":
            return self.test
        else:
            raise ValueError(f"Split '{split}' no válido. Use 'train', 'dev', o 'test'")
    
    def get_texts_and_labels(self, split: str) -> Tuple[List[str], List[List[str]]]:
        """
        Obtiene textos y labels de un split en formato lista.
        
        Args:
            split: 'train', 'dev', o 'test'
            
        Returns:
            Tupla de (textos, labels) donde labels es una lista de listas de etiquetas
        """
        df = self.get_split(split)
        texts = df["input_text"].tolist()
        labels = [label_array.tolist() for label_array in df["labels"]]
        return texts, labels
    
    def get_label_distribution(self, split: str) -> Dict[str, int]:
        """
        Calcula la distribución de etiquetas en un split.
        
        Args:
            split: 'train', 'dev', o 'test'
            
        Returns:
            Diccionario con el conteo de cada etiqueta
        """
        _, labels = self.get_texts_and_labels(split)
        distribution = {label: 0 for label in self.labels}
        
        for label_list in labels:
            for label in label_list:
                distribution[label] += 1
                
        return distribution
    
    def get_stats(self) -> Dict:
        """
        Obtiene estadísticas generales del dataset.
        
        Returns:
            Diccionario con estadísticas del dataset
        """
        return {
            "n_labels": len(self.labels),
            "labels": self.labels,
            "train_size": len(self.train),
            "dev_size": len(self.dev),
            "test_size": len(self.test),
            "total_size": len(self.train) + len(self.dev) + len(self.test)
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"MultiLabelDataset(\n"
            f"  Labels: {stats['n_labels']}\n"
            f"  Train: {stats['train_size']} samples\n"
            f"  Dev: {stats['dev_size']} samples\n"
            f"  Test: {stats['test_size']} samples\n"
            f")"
        )
    

dataset = MultiLabelDataset(data_dir="data/multilabel_banrep")
print(dataset)

