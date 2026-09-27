#!/usr/bin/env python3
"""
convert_txt_to_jsonl.py: Parses training_data.txt (Categories 01-12) and
generates clean ChatML formatted records in dataset.jsonl.
"""

import json
import os
import re

SYSTEM_PROMPT = (
    "You are a clinical AI assistant that parses doctor-patient consultations into structured "
    "database actions (ADD_PATIENT, LOG_VISIT, ADD_CONDITION, ADD_MEDICATION, UPDATE_MEDICATION, "
    "ADD_ALLERGY, ADD_TREATMENT) followed by a conversational SAY: statement."
)


def parse_training_txt(input_file: str = "training_data.txt"):
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by [CATEGORY]:
    raw_blocks = re.split(r"\[CATEGORY\]:", content)
    samples = []

    for block in raw_blocks:
        block = block.strip()
        if not block or "USER:" not in block or "ASSISTANT:" not in block:
            continue

        try:
            category_line, rest = block.split("\n", 1)
            category = category_line.strip()

            user_part, assistant_part = rest.split("ASSISTANT:", 1)
            user_text = user_part.replace("USER:", "").strip()
            assistant_text = assistant_part.strip()

            sample = {
                "category": category,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ],
            }
            samples.append(sample)
        except Exception as e:
            print(f"[!] Warning parsing block: {e}")

    return samples


def generate_jsonl(input_file: str = "training_data.txt", output_file: str = "dataset.jsonl"):
    samples = parse_training_txt(input_file)
    print(f"[*] Parsed {len(samples)} examples across 12 categories from {input_file}.")

    with open(output_file, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps({"messages": item["messages"]}) + "\n")

    print(f"[+] Successfully wrote {len(samples)} training records to: {output_file}")


if __name__ == "__main__":
    generate_jsonl()
