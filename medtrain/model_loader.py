#!/usr/bin/env python3
"""
model_loader.py: Utilities for downloading, loading, and configuring AI models
and LoRA adapters for fine-tuning and export.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


def get_default_device():
    """Detects available acceleration device (CUDA / MPS / CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(
    model_id_or_path: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list = None,
    torch_dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = True,
):
    """
    Loads tokenizer and model. If use_lora is True, wraps the model in a PEFT
    LoRA configuration for efficient fine-tuning.
    """
    device = get_default_device()
    print(f"[*] Loading tokenizer for '{model_id_or_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id_or_path,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[*] Loading base model '{model_id_or_path}' on device={device} with dtype={torch_dtype}...")
    device_map = "auto" if device == "cuda" else None

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )

    if device == "cpu" and device_map is None:
        model = model.to(torch.float32)

    if use_lora:
        if target_modules is None:
            # Standard attention & MLP projection layers for LLaMA/SmolLM2/Qwen
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

        print(f"[*] Configuring LoRA (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        print("[*] Trainable parameters:")
        model.print_trainable_parameters()

    return model, tokenizer


def merge_and_save_lora(
    base_model_id_or_path: str,
    adapter_path: str,
    output_dir: str,
    trust_remote_code: bool = True,
):
    """
    Merges fine-tuned LoRA adapter weights back into base model and saves
    the full standalone model in Hugging Face / SafeTensors format.
    """
    print(f"[*] Merging LoRA adapters from '{adapter_path}' into base '{base_model_id_or_path}'...")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=trust_remote_code)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id_or_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )

    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = peft_model.merge_and_unload()

    print(f"[*] Saving merged weights to '{output_dir}'...")
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    print(f"[+] Merged model and tokenizer successfully saved to: {output_dir}")
    return output_dir
