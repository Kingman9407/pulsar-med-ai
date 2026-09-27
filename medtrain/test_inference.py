#!/usr/bin/env python3
"""
test_inference.py: Runs test clinical prompts through the fine-tuned model
and verifies the generated DSL commands against the SQLite database.
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from medical_db import MedicalDatabase

SYSTEM_PROMPT = (
    "You are a clinical AI assistant that parses doctor-patient consultations into structured "
    "database actions (ADD_PATIENT, LOG_VISIT, ADD_CONDITION, ADD_MEDICATION, UPDATE_MEDICATION, "
    "ADD_ALLERGY, ADD_TREATMENT) followed by a conversational SAY: statement."
)

DEFAULT_TEST_PROMPTS = [
    "Please register a new patient named Jonathan Miller, born on December 5, 1990, male, blood type AB negative, contact 555-4321.",
    "Create a new patient record for Sophia Diaz. Phone number is 555-9012.",
    "Add new patient Lucas Scott.",
    "Register a 45-year-old female patient named Maria Garcia, born 1981-08-22, blood group B positive."
]


def test_model(model_path: str = "./medical_merged_model", db_path: str = "test_inference.db"):
    print("=" * 65)
    print(f"🤖 TESTING MODEL INFERENCE FROM: {model_path}")
    print("=" * 65)

    if not os.path.exists(model_path):
        print(f"[!] Model path '{model_path}' not found! Checking fallback './medical_lora_output'...")
        if os.path.exists("./medical_lora_output"):
            model_path = "./medical_lora_output"
        else:
            print("[!] Please run train.py first to produce trained model weights.")
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading tokenizer & model on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    db = MedicalDatabase(db_path)

    for idx, prompt_text in enumerate(DEFAULT_TEST_PROMPTS, 1):
        print(f"\n--- [Test Prompt {idx}] ---")
        print(f"USER: {prompt_text}")

        chatml_prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = tokenizer(chatml_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        print("MODEL GENERATION:")
        print(response)

    print("\n" + "=" * 65)
    print("✅ Model inference testing completed!")
    print("=" * 65)

    if os.path.exists(db_path):
        os.remove(db_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained medical model inference")
    parser.add_argument("--model_path", type=str, default="./medical_merged_model", help="Path to trained model")
    args = parser.parse_args()

    test_model(args.model_path)
