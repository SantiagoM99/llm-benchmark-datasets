"""
Prompt template for single-label (multinomial) classification tasks.
"""
from typing import List, Optional
from prompts.base_prompt import BasePromptTemplate


class MultinomialPromptTemplate(BasePromptTemplate):
    """
    Prompt template for single-label classification tasks.
    """
    
    def __init__(
        self, 
        available_labels: List[str], 
        language: str = "es",
        examples: Optional[List[dict]] = None
    ):
        """
        Initialize the single-label prompt template.
        
        Args:
            available_labels: List of possible labels
            language: Prompt language ('es' or 'en')
            examples: Optional few-shot examples [{"text": ..., "label": ...}, ...]
        """
        super().__init__(language=language)
        self.available_labels = available_labels
        self.examples = examples or []
    
    @property
    def is_few_shot(self) -> bool:
        return len(self.examples) > 0
    
    def _format_examples(self) -> str:
        """Format few-shot examples."""
        if not self.examples:
            return ""
        
        formatted = []
        for ex in self.examples:
            formatted.append(f"Texto: {ex['text']}\nCategoría: {ex['label']}")
        
        return "\n\n".join(formatted)
    
    def create_prompt(self, text: str, **kwargs) -> str:
        """
        Create a prompt for single-label classification.
        """
        labels_str = "\n".join(f"- {label}" for label in self.available_labels)
        
        # Few-shot section
        few_shot_section = ""
        if self.is_few_shot:
            few_shot_section = f"\n\nEjemplos:\n\n{self._format_examples()}\n\n---"
        
        if self.language == "es":
            prompt = f"""Eres un clasificador de textos académicos. Tu tarea es clasificar el siguiente texto en exactamente UNA de las categorías disponibles.

Categorías disponibles:
{labels_str}

Instrucciones:
- Responde ÚNICAMENTE con el nombre exacto de la categoría
- Debes elegir exactamente UNA categoría
- No agregues explicaciones ni texto adicional{few_shot_section}

Texto a clasificar:
{text}

Categoría:"""
        else:
            prompt = f"""You are an academic text classifier. Your task is to classify the following text into exactly ONE of the available categories.

Available categories:
{labels_str}

Instructions:
- Respond ONLY with the exact category name
- You must choose exactly ONE category
- Do not add explanations or additional text{few_shot_section}

Text to classify:
{text}

Category:"""
        
        return prompt
    
    def parse_response(self, response: str) -> str:
        """Parse the model's response to extract predicted label."""
        response = response.strip().lower()
        
        for label in self.available_labels:
            if label.lower() == response:
                return label
            if label.lower() in response:
                return label
        
        return response
    
    def __repr__(self) -> str:
        return (
            f"MultinomialPromptTemplate(\n"
            f"  n_labels={len(self.available_labels)},\n"
            f"  language='{self.language}',\n"
            f"  few_shot={self.is_few_shot} ({len(self.examples)} examples)\n"
            f")"
        )