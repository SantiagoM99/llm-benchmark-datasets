"""
OpenAI and Azure OpenAI LLM wrapper.
"""
import os
from typing import List
from openai import OpenAI, AzureOpenAI, BadRequestError
from models.base_llm import BaseLLM
import time


class OpenAILLM(BaseLLM):
    """Wrapper for OpenAI models (direct API, not Azure)."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", **kwargs):
        super().__init__(model_name, **kwargs)
        
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except BadRequestError as e:
            print(f"Content filter triggered, returning 'FILTERED'")
            return "FILTERED"
        except Exception as e:
            print(f"Error: {e}")
            return "ERROR"
    
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
            time.sleep(0.5)
        return responses
    
    def __repr__(self) -> str:
        return f"OpenAILLM(model_name='{self.model_name}')"


class AzureOpenAILLM(BaseLLM):
    """Wrapper for Azure OpenAI models."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", **kwargs):
        super().__init__(model_name, **kwargs)
        
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2025-03-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except BadRequestError as e:
            print(f"Content filter triggered, returning 'FILTERED'")
            return "FILTERED"
    
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
        return f"AzureOpenAILLM(model_name='{self.model_name}')"