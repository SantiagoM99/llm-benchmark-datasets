from typing import List, Tuple, Optional
from models.base_multilabel import BaseMultiLabelModel
from models.base_llm import BaseLLM
from prompts.multilabel_prompt import MultiLabelPromptTemplate

class LLMMultiLabelModel(BaseMultiLabelModel):
    """
    Multi-label classification model using any LLM.
    
    This class combines:
    - A base LLM (HuggingFace, OpenAI, etc.)
    - A prompt template for multi-label classification
    - The logic to predict multiple labels
    """
    
    def __init__(
        self, 
        llm: BaseLLM,
        available_labels: List[str],
        prompt_template: Optional[MultiLabelPromptTemplate] = None,
        batch_size: int = 8
    ):
        """
        Initialize the multi-label classification model.
        
        Args:
            llm: Instance of an LLM (HuggingFace, OpenAI, etc.)
            available_labels: List of possible labels
            prompt_template: Custom prompt template (optional)
            batch_size: Batch size for processing
        """
        super().__init__(model_name=f"{llm.model_name}_multilabel")
        self.llm = llm
        self.available_labels = available_labels
        self.prompt_template = prompt_template or MultiLabelPromptTemplate(
            available_labels=available_labels
        )
        self.batch_size = batch_size
    
    def predict(self, texts: List[str]) -> List[List[str]]:
        """
        Predict labels for a list of texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of lists of predicted labels
        """
        # Create prompts for all texts
        prompts = [self.prompt_template.create_prompt(text=text) for text in texts]
        
        # Generate responses in batch
        responses = self.llm.batch_generate(
            prompts=prompts,
            temperature=0.0,  # Deterministic for classification
            max_tokens=100    # Labels are short
        )
        
        # Parse responses
        predictions = [
            self.prompt_template.parse_response(response) 
            for response in responses
        ]
        
        return predictions
    
    def predict_with_scores(
        self, 
        texts: List[str]
    ) -> List[Tuple[List[str], List[float]]]:
        """
        Predict labels with confidence scores.
        
        For basic LLMs without logprobs, we return score 1.0 for all predictions.
        Subclasses can override this if they have access to probabilities.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of tuples (labels, scores)
        """
        predictions = self.predict(texts)
        return [(labels, [1.0] * len(labels)) for labels in predictions]
    
    def __repr__(self) -> str:
        return (
            f"LLMMultiLabelModel(\n"
            f"  llm={self.llm.model_name},\n"
            f"  n_labels={len(self.available_labels)},\n"
            f"  batch_size={self.batch_size}\n"
            f")"
        )