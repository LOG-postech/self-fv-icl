"""
Model loading and configuration utilities.
"""

import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LlamaForCausalLM, AutoConfig


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    # Random seed
    random.seed(seed)

    # Numpy seed
    np.random.seed(seed)

    # Torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # os seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_model(args):
    """Load model and tokenizer based on model type specified in args.
    
    Args:
        args: Arguments object containing model configuration
        
    Returns:
        tuple: (tokenizer, model)
    """
    if "Qwen" in args.model or "internlm" in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = min(tokenizer.model_max_length, 2048)
    
    if "falcon" in args.model or "deepseek" in args.model:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
        if "deepseek" in args.model:
            model.generation_config = GenerationConfig.from_pretrained(args.model)
            model.generation_config.pad_token_id = model.generation_config.eos_token_id
    elif "Llama" in args.model:
        model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    elif "Mistral" in args.model or "mpt" in args.model or "Mixtral" in args.model:
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float16)
    elif "Qwen" in args.model:
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    elif "gpt-j-6b" in args.model:
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float16)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise NotImplementedError("Not supported model.")
    
    return tokenizer, model
