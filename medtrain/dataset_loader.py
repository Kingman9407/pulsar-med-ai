#!/usr/bin/env python3
"""
dataset_loader.py: Loads and formats JSONL datasets for supervised fine-tuning.
Supports OpenAI messages format, Alpaca instruction format, and prompt/response pairs.
"""

import json
import os
from datasets import Dataset, load_dataset


DEFAULT_SAMPLE_PROMPTS = [
    {
        "messages": [
            {"role": "system", "content": "You are an empathetic, expert AI Medical Assistant."},
            {"role": "user", "content": "What are the common early symptoms of Type 2 Diabetes?"},
            {"role": "assistant", "content": "Common early symptoms of Type 2 Diabetes include increased thirst (polydipsia), frequent urination (polyuria), unexplained fatigue, blurred vision, slow-healing sores or cuts, and tingling or numbness in the hands or feet. If you notice these symptoms, it is recommended to get a fasting blood glucose or HbA1c test from a medical provider."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are an empathetic, expert AI Medical Assistant."},
            {"role": "user", "content": "How should a mild first-degree burn be treated immediately?"},
            {"role": "assistant", "content": "For a mild first-degree burn: 1) Cool the burn immediately under cool (not cold or icy) running water for 10 to 15 minutes. 2) Gently clean the area with mild soap. 3) Apply petroleum jelly or aloe vera gel to soothe and protect the skin. 4) Cover loosely with a sterile bandage if needed. Do not pop blisters or apply butter or ice, as this can worsen tissue damage."}
        ]
    }
]


def format_chatml_record(messages):
    """Formats list of message dicts into standard ChatML text."""
    formatted = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        formatted.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(formatted)


def parse_jsonl_line(line_str, tokenizer=None):
    """
    Parses a single JSON line into a formatted training string.
    Supports:
      1. {"messages": [{"role": "...", "content": "..."}]}
      2. {"instruction": "...", "input": "...", "output": "..."}
      3. {"prompt": "...", "response": "..."} or {"question": "...", "answer": "..."}
      4. {"Description": "...", "Doctor": "..."}
    """
    data = json.loads(line_str.strip())

    # Case 1: Chat messages format
    if "messages" in data and isinstance(data["messages"], list):
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            try:
                return tokenizer.apply_chat_template(data["messages"], tokenize=False)
            except Exception:
                pass
        return format_chatml_record(data["messages"])

    # Case 2: Alpaca format
    if "instruction" in data and "output" in data:
        system = "You are a professional AI Medical Assistant."
        instruction = data.get("instruction", "")
        context = data.get("input", "")
        user_query = f"{instruction}\n\nContext: {context}" if context else instruction
        assistant_resp = data.get("output", "")
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_query.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_resp.strip()}<|im_end|>"
        )

    # Case 3: Prompt / Response format
    prompt = data.get("prompt") or data.get("question") or data.get("Description") or ""
    response = data.get("response") or data.get("answer") or data.get("Doctor") or ""
    if prompt and response:
        system = "You are a professional AI Medical Assistant."
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n{response.strip()}<|im_end|>"
        )

    # Fallback: Raw text field if present
    if "text" in data:
        return data["text"]

    raise ValueError(f"Unrecognized record format in line: {line_str[:80]}...")


def load_and_prepare_dataset(jsonl_path: str = "dataset.jsonl", tokenizer=None):
    """
    Loads JSONL dataset from path. If file is empty, logs advice and
    provides starter samples or pulls from Hugging Face public medical dataset.
    """
    records = []

    if os.path.exists(jsonl_path) and os.path.getsize(jsonl_path) > 0:
        print(f"[*] Reading training samples from: {jsonl_path}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    formatted_text = parse_jsonl_line(line, tokenizer=tokenizer)
                    records.append({"text": formatted_text})
                except Exception as e:
                    print(f"[!] Warning on line {idx + 1}: {e}")

        print(f"[*] Loaded {len(records)} samples from '{jsonl_path}'.")

    if not records:
        print(f"[!] Notice: '{jsonl_path}' is currently empty (0 bytes).")
        print("[*] To use your own data, append JSON lines like this:")
        print('    {"messages": [{"role": "user", "content": "Doctor, what is asthma?"}, {"role": "assistant", "content": "Asthma is a chronic condition..."}]}')
        print("[*] Loading starter medical consultation dataset (ruslanmv/ai-medical-chatbot) as a fallback...")
        try:
            hf_data = load_dataset("ruslanmv/ai-medical-chatbot", split="train[:500]")
            for item in hf_data:
                desc = item.get("Description", "")
                doc = item.get("Doctor", "")
                if desc and doc:
                    prompt = (
                        f"<|im_start|>system\nYou are an empathetic, expert AI Medical Assistant.<|im_end|>\n"
                        f"<|im_start|>user\n{desc.strip()}<|im_end|>\n"
                        f"<|im_start|>assistant\n{doc.strip()}<|im_end|>"
                    )
                    records.append({"text": prompt})
            print(f"[*] Loaded {len(records)} starter records from public dataset.")
        except Exception as e:
            print(f"[!] Could not download public dataset: {e}. Using built-in default prompts.")
            for sample in DEFAULT_SAMPLE_PROMPTS:
                records.append({"text": format_chatml_record(sample["messages"])})

    hf_dataset = Dataset.from_list(records)
    print(f"[+] Final prepared dataset contains {len(hf_dataset)} examples.")
    return hf_dataset


if __name__ == "__main__":
    ds = load_and_prepare_dataset("dataset.jsonl")
    print("\n--- Example formatted record ---")
    print(ds[0]["text"][:300])
