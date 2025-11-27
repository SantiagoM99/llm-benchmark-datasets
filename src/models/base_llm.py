# src/models/base_llm.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseLLM(ABC):
    """
    Abstract base class for any LLM, independent of the task.
    
    This class defines the common interface for all language models,
    regardless of whether they're from HuggingFace, OpenAI, Anthropic, etc.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the LLM.
        
        Args:
            model_name: Name/identifier of the model
            **kwargs: Additional model-specific configuration
        """
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    def generate(
        self, 
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate a response given a prompt.
        
        Args:
            prompt: The prompt to send to the model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Model response as string
        """
        pass
    
    @abstractmethod
    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts
            max_tokens: Maximum number of tokens to generate per prompt
            temperature: Sampling temperature
            **kwargs: Additional model-specific parameters
            
        Returns:
            List of responses
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"