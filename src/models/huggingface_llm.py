# src/models/huggingface_llm.py
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from models.base_llm import BaseLLM


class HuggingFaceLLM(BaseLLM):
    """
    Generic wrapper for HuggingFace models.
    """
    
    def __init__(
        self, 
        model_name: str,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        use_flash_attention: bool = False,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch_dtype or torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        model_kwargs = {
            "quantization_config": quantization_config,
            "torch_dtype": torch_dtype or torch.float16,
            "trust_remote_code": trust_remote_code,
            "device_map": "cuda:0",
            **kwargs
        }
        
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")
        if load_in_8bit:
            quant_str = "8-bit"
        elif load_in_4bit:
            quant_str = "4-bit"
        else:
            quant_str = "None"
        print(f"Quantization: {quant_str}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        if not quantization_config:
            self.model.to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def generate(
        self, 
        prompt: str,
        max_tokens: int = 512,
        top_p: float = 0.9,
        do_sample: bool = False,
        **kwargs
    ) -> str:
        """Generate a response using the model's chat template."""
        
        # Usar chat template si el modelo lo soporta
        messages = [{"role": "user", "content": prompt}]
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = False,
        batch_size: int = 1,  # Cambiado a 1 para evitar problemas
        **kwargs
    ) -> List[str]:
        """Generate responses for multiple prompts."""
        responses = []
        
        for prompt in prompts:
            response = self.generate(
                prompt,
                max_tokens=max_tokens,
                do_sample=do_sample,
                **kwargs
            )
            responses.append(response)
        
        return responses
    
    def __repr__(self) -> str:
        return (
            f"HuggingFaceLLM(\n"
            f"  model_name='{self.model_name}',\n"
            f"  device='{self.device}',\n"
            f"  dtype={self.model.dtype}\n"
            f")"
        )