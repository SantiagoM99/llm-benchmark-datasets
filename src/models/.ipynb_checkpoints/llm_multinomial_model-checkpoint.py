"""
Single-label classification model using LLMs with Weave tracking.
"""
from typing import List, Optional

import weave

from models.base_multinomial import BaseMultinomialModel
from models.base_llm import BaseLLM
from prompts.multinomial_prompt import MultinomialPromptTemplate


class LLMMultinomialModel(BaseMultinomialModel):
    """
    Single-label classification model using any LLM.
    
    This class combines:
    - A base LLM (HuggingFace, OpenAI, etc.)
    - A prompt template for single-label classification
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        available_labels: List[str],
        prompt_template: Optional[MultinomialPromptTemplate] = None,
        language: str = "es"
    ):
        """
        Initialize the single-label classification model.
        
        Args:
            llm: Instance of an LLM (HuggingFace, OpenAI, etc.)
            available_labels: List of possible labels
            prompt_template: Custom prompt template (optional)
            language: Language for prompts ('es' or 'en')
        """
        super().__init__(model_name=f"{llm.model_name}_multinomial")
        self.llm = llm
        self.available_labels = available_labels
        self.prompt_template = prompt_template or MultinomialPromptTemplate(
            available_labels=available_labels,
            language=language
        )
    
    @weave.op()
    def predict_single(self, text: str) -> str:
        """
        Predict label for a single text.
        
        Args:
            text: Text to classify
            
        Returns:
            Predicted label
        """
        prompt = self.prompt_template.create_prompt(text=text)
        response = self.llm.generate(prompt, max_tokens=50, temperature=0.0)
        prediction = self.prompt_template.parse_response(response)
        
        return prediction
    
    @weave.op()
    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict labels for a list of texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of predicted labels
        """
        prompts = [self.prompt_template.create_prompt(text=text) for text in texts]
        
        responses = self.llm.batch_generate(
            prompts=prompts,
            max_tokens=50,
            temperature=0.0
        )
        
        predictions = [
            self.prompt_template.parse_response(response)
            for response in responses
        ]
        
        return predictions
    
    def __repr__(self) -> str:
        return (
            f"LLMMultinomialModel(\n"
            f"  llm={self.llm.model_name},\n"
            f"  n_labels={len(self.available_labels)}\n"
            f")"
        )