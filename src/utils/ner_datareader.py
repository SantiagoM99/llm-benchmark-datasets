"""
DataLoader for NER (Named Entity Recognition) datasets in CoNLL format.
"""
from pathlib import Path
from typing import Dict, List, Tuple


class NERDataset:
    """
    Clase para cargar y gestionar datasets NER en formato CoNLL.
    
    Formato CoNLL esperado:
    - Una palabra por línea con formato: TOKEN TAG
    - Líneas vacías separan oraciones
    - Esquema de etiquetado: BIO (B-ENTITY, I-ENTITY, O)
    """
    
    def __init__(self, data_dir: str):
        """
        Inicializa el dataset cargando los archivos CoNLL.
        
        Args:
            data_dir: Path al directorio que contiene train.conll, dev.conll, test.conll
        """
        self.data_dir = Path(data_dir)
        
        self.train = self._load_conll("train.conll")
        self.dev = self._load_conll("dev.conll")
        self.test = self._load_conll("test.conll")
        
        self.entity_types = self._extract_entity_types()
    
    def _load_conll(self, filename: str) -> List[List[Tuple[str, str]]]:
        """
        Carga un archivo CoNLL.
        
        Returns:
            Lista de oraciones, donde cada oración es una lista de tuplas (token, tag)
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            return None
        
        sentences = []
        current_sentence = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if line == "" or line.startswith("-DOCSTART-"):
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        token = parts[0]
                        tag = parts[-1]  # Última columna es el tag NER
                        current_sentence.append((token, tag))
            
            # Agregar última oración si existe
            if current_sentence:
                sentences.append(current_sentence)
        
        return sentences
    
    def _extract_entity_types(self) -> List[str]:
        """Extrae todos los tipos de entidades únicos del dataset."""
        entity_types = set()
        
        for split in [self.train, self.dev, self.test]:
            if split is None:
                continue
            for sentence in split:
                for token, tag in sentence:
                    if tag != "O":
                        # Extraer tipo de entidad (quitar B- o I-)
                        entity_type = tag.split("-")[-1] if "-" in tag else tag
                        entity_types.add(entity_type)
        
        return sorted(list(entity_types))
    
    def get_split(self, split: str) -> List[List[Tuple[str, str]]]:
        """
        Obtiene un split específico del dataset.
        
        Args:
            split: 'train', 'dev', o 'test'
            
        Returns:
            Lista de oraciones con sus tags
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
    
    def get_sentences_and_labels(self, split: str) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Obtiene tokens y labels separados.
        
        Args:
            split: 'train', 'dev', o 'test'
            
        Returns:
            Tupla de (lista de oraciones como tokens, lista de oraciones como tags)
        """
        data = self.get_split(split)
        sentences = [[token for token, tag in sent] for sent in data]
        labels = [[tag for token, tag in sent] for sent in data]
        return sentences, labels
    
    def get_text_sentences(self, split: str) -> List[str]:
        """
        Obtiene oraciones como strings.
        
        Args:
            split: 'train', 'dev', o 'test'
            
        Returns:
            Lista de oraciones como strings
        """
        sentences, _ = self.get_sentences_and_labels(split)
        return [" ".join(tokens) for tokens in sentences]
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas del dataset."""
        def count_entities(data):
            if data is None:
                return 0, 0
            n_sentences = len(data)
            n_entities = sum(
                1 for sent in data for token, tag in sent if tag.startswith("B-")
            )
            return n_sentences, n_entities
        
        train_sent, train_ent = count_entities(self.train)
        dev_sent, dev_ent = count_entities(self.dev)
        test_sent, test_ent = count_entities(self.test)
        
        return {
            "entity_types": self.entity_types,
            "n_entity_types": len(self.entity_types),
            "train_sentences": train_sent,
            "train_entities": train_ent,
            "dev_sentences": dev_sent,
            "dev_entities": dev_ent,
            "test_sentences": test_sent,
            "test_entities": test_ent,
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"NERDataset(\n"
            f"  Entity types: {stats['n_entity_types']} {stats['entity_types']}\n"
            f"  Train: {stats['train_sentences']} sentences, {stats['train_entities']} entities\n"
            f"  Dev: {stats['dev_sentences']} sentences, {stats['dev_entities']} entities\n"
            f"  Test: {stats['test_sentences']} sentences, {stats['test_entities']} entities\n"
            f")"
        )


class EconIEDataset(NERDataset):
    """Dataset específico para Econ-IE NER."""
    
    def __init__(self, data_dir: str = "data/ner_econ_ie"):
        super().__init__(data_dir=data_dir)