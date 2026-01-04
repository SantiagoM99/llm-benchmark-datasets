"""kNN retriever for NER few-shot examples."""
from typing import List, Dict, Optional
import numpy as np


class KNNRetriever:
    """Sentence-level kNN retriever using sentence-transformers."""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.examples: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def build_index(self, examples: List[Dict]) -> None:
        """Build embedding index from training examples."""
        self.examples = examples
        sentences = [ex["text"] for ex in examples]
        self.embeddings = self.model.encode(sentences, normalize_embeddings=True)
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve k most similar examples."""
        query_emb = self.model.encode([query], normalize_embeddings=True)
        similarities = np.dot(self.embeddings, query_emb.T).flatten()
        top_k = np.argsort(similarities)[-k:][::-1]
        return [self.examples[i] for i in top_k]
    
    def retrieve_balanced(self, query: str, k: int, entity_types: List[str]) -> List[Dict]:
        """Retrieve k examples with entity type coverage."""
        query_emb = self.model.encode([query], normalize_embeddings=True)
        similarities = np.dot(self.embeddings, query_emb.T).flatten()
        
        # Build type index
        type_indices = {t: [] for t in entity_types}
        for idx, ex in enumerate(self.examples):
            for ent in ex.get("entities", []):
                t = ent.get("type", "").lower()
                if t in type_indices and idx not in type_indices[t]:
                    type_indices[t].append(idx)
        
        selected = set()
        # First: one per type
        for t in entity_types:
            if type_indices[t]:
                best_idx = max(type_indices[t], key=lambda i: similarities[i])
                selected.add(best_idx)
                if len(selected) >= k:
                    break
        
        # Fill rest with top similar
        if len(selected) < k:
            for idx in np.argsort(similarities)[::-1]:
                if idx not in selected:
                    selected.add(idx)
                    if len(selected) >= k:
                        break
        
        result = sorted(selected, key=lambda i: similarities[i], reverse=True)
        return [self.examples[i] for i in result[:k]]