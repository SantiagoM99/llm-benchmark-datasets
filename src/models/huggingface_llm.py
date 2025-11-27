# src/models/huggingface_llm.py
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from models.base_llm import BaseLLM


class HuggingFaceLLM(BaseLLM):
    """
    Generic wrapper for HuggingFace models.
    
    Supports various models (Llama, Gemma, Qwen, etc.) with options for
    quantization and different hardware configurations.
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
        """
        Initialize the HuggingFace LLM.
        
        Args:
            model_name: HuggingFace model identifier (e.g., 'meta-llama/Llama-3.1-8B-Instruct')
            device: Device to load model on ('cuda', 'cpu', or None for auto-detect)
            load_in_8bit: Load model with 8-bit quantization
            load_in_4bit: Load model with 4-bit quantization
            torch_dtype: Torch dtype for model weights (default: float16)
            trust_remote_code: Trust remote code from model repository
            use_flash_attention: Use Flash Attention 2 for faster inference
            **kwargs: Additional arguments passed to AutoModelForCausalLM.from_pretrained()
        """
        super().__init__(model_name, **kwargs)
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup quantization config
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch_dtype or torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Model loading kwargs
        model_kwargs = {
            "quantization_config": quantization_config,
            "torch_dtype": torch_dtype or torch.float16,
            "trust_remote_code": trust_remote_code,
            "device_map": "auto" if quantization_config else None,
            **kwargs
        }
        
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Quantization: {'8-bit' if load_in_8bit else '4-bit' if load_in_4bit else 'None'}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Move model to device if not using quantization
        if not quantization_config:
            self.model.to(self.device)
        
        # Setup pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Set model to eval mode
        self.model.eval()
        
        print(f"Model loaded successfully!")
    
    def generate(
        self, 
        prompt: str,
        max_tokens: int = 512,
        top_p: float = 0.9,
        do_sample: bool = False,
        **kwargs
    ) -> str:
        """
        Generate a response given a prompt.
        
        Args:
            prompt: The prompt to send to the model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text as string
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode only the new tokens (without the prompt)
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
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts
            max_tokens: Maximum number of tokens to generate per prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            batch_size: Number of prompts to process simultaneously
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        responses = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=2048  # Prevent extremely long prompts
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Decode each output
            for j, output in enumerate(outputs):
                # Find where the prompt ends
                prompt_length = inputs['input_ids'][j].shape[0]
                response = self.tokenizer.decode(
                    output[prompt_length:],
                    skip_special_tokens=True
                )
                responses.append(response.strip())
        
        return responses
    
    def __repr__(self) -> str:
        return (
            f"HuggingFaceLLM(\n"
            f"  model_name='{self.model_name}',\n"
            f"  device='{self.device}',\n"
            f"  dtype={self.model.dtype}\n"
            f")"
        )