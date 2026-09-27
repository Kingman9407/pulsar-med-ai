#!/usr/bin/env python3
"""
train.py: Main training pipeline.
Fine-tunes the base model on dataset.jsonl using LoRA, merges the adapter,
and converts the resulting checkpoint into model.gguf for Flutter on-device inference.
"""

import argparse
import os
import sys
import torch
from transformers import TrainingArguments
from trl import SFTTrainer

# Local imports from medtrain module
from model_loader import load_model_and_tokenizer, merge_and_save_lora
from dataset_loader import load_and_prepare_dataset
from convert_to_gguf import convert_model


def run_training(
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    dataset_path: str = "dataset.jsonl",
    output_dir: str = "./medical_lora_output",
    merged_dir: str = "./medical_merged_model",
    gguf_output: str = "./model.gguf",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 1024,
    export_gguf: bool = True,
):
    print("=" * 60)
    print("🩺 MEDICAL MODEL FINE-TUNING & GGUF EXPORT PIPELINE")
    print("=" * 60)
    print(f"Base Model:    {model_id}")
    print(f"Dataset Path:  {dataset_path}")
    print(f"LoRA Output:   {output_dir}")
    print(f"Merged Dir:    {merged_dir}")
    print(f"GGUF Output:   {gguf_output}")
    print(f"Epochs:        {epochs}")
    print(f"Batch Size:    {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print("=" * 60)

    # 1. Load Model & Tokenizer with LoRA
    model, tokenizer = load_model_and_tokenizer(
        model_id_or_path=model_id,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    # 2. Load Dataset from dataset.jsonl
    dataset = load_and_prepare_dataset(jsonl_path=dataset_path, tokenizer=tokenizer)

    # 3. Configure Training Arguments
    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 16 // batch_size),
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=use_fp16,
        report_to="none",
    )

    # 4. Initialize SFTTrainer
    print("[*] Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    # 5. Execute Training
    print("[*] Starting model training...")
    trainer.train()
    print("[+] Training completed successfully!")

    # 6. Save LoRA Adapters
    print(f"[*] Saving fine-tuned LoRA weights to: {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 7. Merge LoRA into Base Model
    print("[*] Merging LoRA adapters with base model...")
    merge_and_save_lora(
        base_model_id_or_path=model_id,
        adapter_path=output_dir,
        output_dir=merged_dir,
    )

    # 8. Export to GGUF
    if export_gguf:
        print("[*] Converting merged model to GGUF format...")
        try:
            convert_model(model_dir=merged_dir, out_file=gguf_output)
            print("=" * 60)
            print(f"🎉 All Done! Your model is ready at: {gguf_output}")
            print("You can now copy it to your Flutter app's assets folder.")
            print("=" * 60)
        except Exception as e:
            print(f"[!] GGUF Conversion encountered an error: {e}")
            print(f"Note: You can still run 'python convert_to_gguf.py --model_dir {merged_dir}' separately.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Medical LLM and Export to GGUF")
    parser.add_argument("--model_id", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct", help="Base HF model ID or local directory")
    parser.add_argument("--dataset_path", type=str, default="dataset.jsonl", help="Path to JSONL dataset file")
    parser.add_argument("--output_dir", type=str, default="./medical_lora_output", help="Directory to save LoRA adapters")
    parser.add_argument("--merged_dir", type=str, default="./medical_merged_model", help="Directory to save merged model")
    parser.add_argument("--gguf_output", type=str, default="./model.gguf", help="Output path for GGUF binary")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max token sequence length")
    parser.add_argument("--no_gguf", action="store_true", help="Skip GGUF export")

    args = parser.parse_args()

    run_training(
        model_id=args.model_id,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        merged_dir=args.merged_dir,
        gguf_output=args.gguf_output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        export_gguf=not args.no_gguf,
    )
