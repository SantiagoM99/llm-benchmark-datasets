"""
OpenAI and Azure OpenAI LLM wrapper.
"""
import os
from typing import List
from openai import AzureOpenAI
from models.base_llm import BaseLLM


class OpenAILLM(BaseLLM):
    """Wrapper for Azure OpenAI models."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", **kwargs):
        super().__init__(model_name, **kwargs)
        
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-10-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    
    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> List[str]:
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, max_tokens, temperature, **kwargs)
            responses.append(response)
        return responses
    
    def __repr__(self) -> str:
        return f"OpenAILLM(model_name='{self.model_name}')"