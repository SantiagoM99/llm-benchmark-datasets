# src/prompts/multilabel_prompt.py
from typing import List
from prompts.base_prompt import BasePromptTemplate


class MultiLabelPromptTemplate(BasePromptTemplate):
    """
    Prompt template for multi-label classification tasks.
    
    This template creates prompts that ask the LLM to classify text
    into one or more categories from a predefined set.
    """
    
    def __init__(self, available_labels: List[str], language: str = "es"):
        """
        Initialize the multi-label prompt template.
        
        Args:
            available_labels: List of possible labels
            language: Prompt language ('es' or 'en')
        """
        super().__init__(language=language)
        self.available_labels = available_labels
    
    def create_prompt(self, text: str, **kwargs) -> str:
        """
        Create a prompt for multi-label classification.
        
        Args:
            text: Text to classify
            **kwargs: Additional parameters (unused)
            
        Returns:
            Formatted prompt as string
        """
        labels_str = ", ".join(self.available_labels)
        
        if self.language == "es":
            prompt = f"""Eres un asistente experto en clasificación de documentos económicos.

Categorías disponibles: {labels_str}

Tu tarea es clasificar el siguiente texto en una o más de las categorías anteriores.

Texto a clasificar:
{text}

Instrucciones:
- Responde ÚNICAMENTE con las letras de las categorías aplicables
- Separa múltiples categorías con comas
- No agregues explicaciones ni texto adicional
- Ejemplo de respuesta válida: E, G

Categorías:"""
        else:
            prompt = f"""You are an expert assistant in economic document classification.

Available categories: {labels_str}

Your task is to classify the following text into one or more of the above categories.

Text to classify:
{text}

Instructions:
- Respond ONLY with the letters of applicable categories
- Separate multiple categories with commas
- Do not add explanations or additional text
- Example of valid response: E, G

Categories:"""
        
        return prompt
    
    def parse_response(self, response: str) -> List[str]:
        """
        Parse the model's response to extract predicted labels.
        
        Args:
            response: Raw model response
            
        Returns:
            List of predicted labels
            
        Example:
            >>> parse_response("E, G, H")
            ['E', 'G', 'H']
            >>> parse_response("The categories are: E and G")
            ['E', 'G']
        """
        # Clean the response
        response = response.strip()
        
        # Extract only valid labels
        labels = []
        for token in response.replace(",", " ").split():
            token = token.strip().upper()
            # Remove punctuation
            token = ''.join(c for c in token if c.isalnum())
            if token in self.available_labels:
                labels.append(token)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_labels = []
        for label in labels:
            if label not in seen:
                seen.add(label)
                unique_labels.append(label)
        
        return unique_labels
    
    def __repr__(self) -> str:
        return (
            f"MultiLabelPromptTemplate(\n"
            f"  n_labels={len(self.available_labels)},\n"
            f"  language='{self.language}'\n"
            f")"
        )