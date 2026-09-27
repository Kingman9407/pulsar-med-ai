#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import safetensors.torch
import gguf

def convert():
    src_dir = Path("AI_model/safetenosr")
    dest_dir = Path("AI_model/gguf")
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_file = dest_dir / "model.gguf"

    print(f"[*] Starting conversion from {src_dir} to {out_file}...")

    # 1. Load config.json
    config_path = src_dir / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 2. Initialize GGUF Writer for LLaMA architecture
    writer = gguf.GGUFWriter(str(out_file), "llama")

    # 3. Model Architecture & Hyperparameters
    writer.add_name("SmolLM2-135M-Instruct")
    writer.add_context_length(config.get("max_position_embeddings", 8192))
    writer.add_embedding_length(config.get("hidden_size", 576))
    writer.add_block_count(config.get("num_hidden_layers", 30))
    writer.add_feed_forward_length(config.get("intermediate_size", 1536))
    writer.add_head_count(config.get("num_attention_heads", 9))
    writer.add_head_count_kv(config.get("num_key_value_heads", 3))
    writer.add_layer_norm_rms_eps(config.get("rms_norm_eps", 1e-05))
    writer.add_rope_freq_base(float(config.get("rope_theta", 100000.0)))
    writer.add_rope_dimension_count(64)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    # 4. Chat Template
    chat_template_path = src_dir / "chat_template.jinja"
    if chat_template_path.exists():
        with open(chat_template_path, "r", encoding="utf-8") as f:
            template_content = f.read()
        writer.add_chat_template(template_content)

    # 5. Tokenizer & Vocabulary
    print("[*] Extracting BPE Tokenizer Vocabulary...")
    vocab = gguf.BpeVocab(src_dir)
    tokens = []
    scores = []
    toktypes = []
    for tok, score, ttype in vocab.all_tokens():
        tokens.append(tok)
        scores.append(score)
        toktypes.append(ttype)

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("smollm")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(toktypes)

    # Add BPE merges
    merges_path = src_dir / "merges.txt"
    if merges_path.exists():
        with open(merges_path, "r", encoding="utf-8") as f:
            merges = [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
        writer.add_token_merges(merges)
        print(f"[*] Added {len(merges)} BPE merges.")

    # Special tokens
    special_vocab = gguf.SpecialVocab(src_dir)
    special_vocab.add_to_gguf(writer)
    print(f"[*] Added special tokens: {special_vocab.special_token_ids}")

    # 6. Tensor Mapping and Weights
    safetensor_file = src_dir / "model.safetensors"
    print(f"[*] Loading weights from {safetensor_file}...")
    weights = safetensors.torch.load_file(str(safetensor_file))

    tensor_map = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
    }

    num_layers = config.get("num_hidden_layers", 30)
    for i in range(num_layers):
        tensor_map.update({
            f"model.layers.{i}.input_layernorm.weight": f"blk.{i}.attn_norm.weight",
            f"model.layers.{i}.self_attn.q_proj.weight": f"blk.{i}.attn_q.weight",
            f"model.layers.{i}.self_attn.k_proj.weight": f"blk.{i}.attn_k.weight",
            f"model.layers.{i}.self_attn.v_proj.weight": f"blk.{i}.attn_v.weight",
            f"model.layers.{i}.self_attn.o_proj.weight": f"blk.{i}.attn_output.weight",
            f"model.layers.{i}.post_attention_layernorm.weight": f"blk.{i}.ffn_norm.weight",
            f"model.layers.{i}.mlp.gate_proj.weight": f"blk.{i}.ffn_gate.weight",
            f"model.layers.{i}.mlp.up_proj.weight": f"blk.{i}.ffn_up.weight",
            f"model.layers.{i}.mlp.down_proj.weight": f"blk.{i}.ffn_down.weight",
        })

    added_tensors = 0
    token_embd_tensor = None

    for hf_name, tensor in weights.items():
        if hf_name not in tensor_map:
            print(f"[!] Warning: unmapped tensor {hf_name}, skipping.")
            continue
        gguf_name = tensor_map[hf_name]
        
        # Normalization layers in GGUF/llama.cpp must be kept as F32
        is_norm = "norm" in gguf_name
        if is_norm:
            np_array = tensor.to(torch.float32).numpy()
            writer.add_tensor(gguf_name, np_array, raw_dtype=gguf.GGMLQuantizationType.F32)
        else:
            np_array = tensor.to(torch.float16).numpy()
            writer.add_tensor(gguf_name, np_array, raw_dtype=gguf.GGMLQuantizationType.F16)
        added_tensors += 1

        if gguf_name == "token_embd.weight":
            token_embd_tensor = np_array

    # For tie_word_embeddings, also add output.weight for broad engine compatibility
    if "output.weight" not in [tensor_map.get(k) for k in weights.keys()] and token_embd_tensor is not None:
        writer.add_tensor("output.weight", token_embd_tensor, raw_dtype=gguf.GGMLQuantizationType.F16)
        added_tensors += 1
        print("[*] Added output.weight tied to token_embd.weight.")

    print(f"[*] Registered {added_tensors} tensors. Writing GGUF binary...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"[+] Conversion complete! Saved to {out_file} ({out_size_mb:.2f} MB)")

    # 7. Verification with GGUFReader
    print("[*] Validating generated GGUF with GGUFReader...")
    reader = gguf.GGUFReader(str(out_file))
    print(f"[*] Verified: {len(reader.tensors)} tensors, {len(reader.fields)} metadata fields.")

if __name__ == "__main__":
    convert()
