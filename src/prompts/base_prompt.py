# src/prompts/base_prompt.py
from abc import ABC, abstractmethod
from typing import Any


class BasePromptTemplate(ABC):
    """
    Base class for task-specific prompt templates.
    
    Each task (classification, NER, summarization) will have its own template
    that inherits from this base class.
    """
    
    def __init__(self, language: str = "es"):
        """
        Initialize the template.
        
        Args:
            language: Prompt language ('es' or 'en')
        """
        self.language = language
    
    @abstractmethod
    def create_prompt(self, **kwargs) -> str:
        """
        Create a task-specific prompt.
        
        Args:
            **kwargs: Data needed to build the prompt
                     (varies by task)
            
        Returns:
            Formatted prompt as string
        """
        pass
    
    @abstractmethod
    def parse_response(self, response: str) -> Any:
        """
        Parse the model's response.
        
        Args:
            response: Raw model response
            
        Returns:
            Parsed response in the format expected by the task
            (can be list of strings, dict, etc.)
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(language='{self.language}')"