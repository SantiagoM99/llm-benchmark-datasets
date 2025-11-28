"""
DataLoader for single-label (multinomial) classification datasets.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


class MultinomialDataset:
    """
    Clase para cargar y gestionar datasets de clasificación single-label.
    
    Attributes:
        data_dir: Directorio que contiene los archivos CSV
        train: DataFrame de entrenamiento (opcional)
        dev: DataFrame de desarrollo/validación (opcional)
        test: DataFrame de prueba
        labels: Lista de etiquetas únicas en el dataset
        input_column: Nombre de la columna con el texto
        label_column: Nombre de la columna con la etiqueta
    """
    
    def __init__(
        self, 
        data_dir: str, 
        input_column: str = "input_text",
        label_column: str = "label"
    ):
        """
        Inicializa el dataset cargando los archivos CSV.
        
        Args:
            data_dir: Path al directorio que contiene los CSV
            input_column: Nombre de la columna con el texto de entrada
            label_column: Nombre de la columna con la etiqueta
        """
        self.data_dir = Path(data_dir)
        self.input_column = input_column
        self.label_column = label_column
        
        self.train = self._load_split("train")
        self.dev = self._load_split("dev")
        self.test = self._load_split("test")
        
        self.labels = self._extract_labels()
    
    def _load_split(self, split: str) -> pd.DataFrame:
        """Carga un split si existe."""
        csv_path = self.data_dir / f"{split}.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return None
    
    def _extract_labels(self) -> List[str]:
        """Extrae todas las etiquetas únicas del dataset."""
        all_labels = set()
        for df in [self.train, self.dev, self.test]:
            if df is not None:
                all_labels.update(df[self.label_column].unique())
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
            if self.train is None:
                raise ValueError("Train split not available")
            return self.train
        elif split == "dev":
            if self.dev is None:
                raise ValueError("Dev split not available")
            return self.dev
        elif split == "test":
            if self.test is None:
                raise ValueError("Test split not available")
            return self.test
        else:
            raise ValueError(f"Split '{split}' no válido. Use 'train', 'dev', o 'test'")
    
    def get_texts_and_labels(self, split: str) -> Tuple[List[str], List[str]]:
        """
        Obtiene textos y labels de un split en formato lista.
        
        Args:
            split: 'train', 'dev', o 'test'
            
        Returns:
            Tupla de (textos, labels)
        """
        df = self.get_split(split)
        texts = df[self.input_column].tolist()
        labels = df[self.label_column].tolist()
        return texts, labels
    
    def get_label_distribution(self, split: str) -> Dict[str, int]:
        """
        Calcula la distribución de etiquetas en un split.
        
        Args:
            split: 'train', 'dev', o 'test'
            
        Returns:
            Diccionario con el conteo de cada etiqueta
        """
        df = self.get_split(split)
        return df[self.label_column].value_counts().to_dict()
    
    def get_stats(self) -> Dict:
        """
        Obtiene estadísticas generales del dataset.
        
        Returns:
            Diccionario con estadísticas del dataset
        """
        return {
            "n_labels": len(self.labels),
            "labels": self.labels,
            "train_size": len(self.train) if self.train is not None else 0,
            "dev_size": len(self.dev) if self.dev is not None else 0,
            "test_size": len(self.test) if self.test is not None else 0,
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"MultinomialDataset(\n"
            f"  Labels: {stats['n_labels']} {stats['labels']}\n"
            f"  Train: {stats['train_size']} samples\n"
            f"  Dev: {stats['dev_size']} samples\n"
            f"  Test: {stats['test_size']} samples\n"
            f")"
        )


class AbstractRosarioDataset(MultinomialDataset):
    """Dataset específico para Abstract Rosario."""
    
    def __init__(self, data_dir: str = "data/abstracts_rosario"):
        super().__init__(
            data_dir=data_dir,
            input_column="input_text",
            label_column="facultad"
        )