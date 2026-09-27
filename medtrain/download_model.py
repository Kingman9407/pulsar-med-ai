#!/usr/bin/env python3
"""
download_model.py: Standalone script to download the base model and tokenizer
from Hugging Face (HuggingFaceTB/SmolLM2-135M-Instruct) into the environment cache.
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


def download(model_id: str = DEFAULT_MODEL_ID, save_local: bool = False, local_dir: str = "./base_model"):
    print("=" * 65)
    print(f"📥 DOWNLOADING MODEL FROM HUGGING FACE: {model_id}")
    print("=" * 65)

    print(f"[*] Downloading tokenizer for '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[+] Tokenizer downloaded successfully.")

    print(f"[*] Downloading model weights for '{model_id}'...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    print("[+] Model weights downloaded successfully.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Total Model Parameters: {total_params / 1e6:.1f}M")

    if save_local:
        print(f"[*] Saving files locally to: {local_dir}...")
        os.makedirs(local_dir, exist_ok=True)
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)
        print(f"[+] Saved model files to {local_dir}")
    else:
        print("[+] Model cached in environment for training.")

    print("=" * 65)
    print(f"🎉 Ready to train with {model_id}!")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model from Hugging Face")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face model ID")
    parser.add_argument("--save_local", action="store_true", help="Also save copy in local folder")
    parser.add_argument("--local_dir", type=str, default="./base_model", help="Target folder if saving locally")
    args = parser.parse_args()

    download(args.model_id, args.save_local, args.local_dir)
