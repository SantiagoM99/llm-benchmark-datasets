"""
NER model using LLMs with Weave tracking.
"""
from typing import List, Dict, Optional

import weave

from models.base_ner import BaseNERModel
from models.base_llm import BaseLLM
from prompts.ner_prompt import NERPromptTemplate


class LLMNERModel(BaseNERModel):
    """
    NER model using any LLM.
    
    This class combines:
    - A base LLM (HuggingFace, OpenAI, etc.)
    - A prompt template for NER
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        entity_types: List[str],
        prompt_template: Optional[NERPromptTemplate] = None,
        language: str = "es"
    ):
        """
        Initialize the NER model.
        
        Args:
            llm: Instance of an LLM (HuggingFace, OpenAI, etc.)
            entity_types: List of entity types to extract
            prompt_template: Custom prompt template (optional)
            language: Language for prompts ('es' or 'en')
        """
        super().__init__(model_name=f"{llm.model_name}_ner")
        self.llm = llm
        self.entity_types = entity_types
        self.prompt_template = prompt_template or NERPromptTemplate(
            entity_types=entity_types,
            language=language
        )
    
    @weave.op()
    def predict_single(self, sentence: str) -> List[Dict[str, str]]:
        """
        Extract entities from a single sentence.
        
        Args:
            sentence: Sentence to process
            
        Returns:
            List of entities [{"text": ..., "type": ...}, ...]
        """
        prompt = self.prompt_template.create_prompt(text=sentence)
        response = self.llm.generate(prompt, max_tokens=500, temperature=0.0)
        entities = self.prompt_template.parse_response(response)
        
        return entities
    
    @weave.op()
    def predict(self, sentences: List[str]) -> List[List[Dict[str, str]]]:
        """
        Extract entities from a list of sentences.
        
        Args:
            sentences: List of sentences to process
            
        Returns:
            List of lists of entities
        """
        all_entities = []
        
        for sentence in sentences:
            entities = self.predict_single(sentence)
            all_entities.append(entities)
        
        return all_entities
    
    def __repr__(self) -> str:
        return (
            f"LLMNERModel(\n"
            f"  llm={self.llm.model_name},\n"
            f"  entity_types={self.entity_types}\n"
            f")"
        )