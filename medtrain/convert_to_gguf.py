#!/usr/bin/env python3
"""
convert_to_gguf.py: Converts SafeTensors/Hugging Face model checkpoints to
the GGUF binary format (.gguf) for on-device inference in Flutter / llama.cpp.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
import safetensors.torch
import gguf


def convert_with_native_gguf(src_dir: Path, out_file: Path):
    """Direct conversion using the Python gguf library for SmolLM2 / LLaMA architectures."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Converting from '{src_dir}' to '{out_file}' using native gguf writer...")

    config_path = src_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {src_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Initialize GGUF Writer for LLaMA architecture
    writer = gguf.GGUFWriter(str(out_file), "llama")

    # Hyperparameters
    model_name = config.get("_name_or_path", "SmolLM2-135M-Instruct")
    writer.add_name(model_name)
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

    # Chat Template
    chat_template_path = src_dir / "chat_template.jinja"
    if chat_template_path.exists():
        with open(chat_template_path, "r", encoding="utf-8") as f:
            writer.add_chat_template(f.read())

    # Tokenizer & Vocabulary
    print("[*] Parsing tokenizer vocabulary...")
    vocab = gguf.BpeVocab(src_dir)
    tokens, scores, toktypes = [], [], []
    for tok, score, ttype in vocab.all_tokens():
        tokens.append(tok)
        scores.append(score)
        toktypes.append(ttype)

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("smollm")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(toktypes)

    merges_path = src_dir / "merges.txt"
    if merges_path.exists():
        with open(merges_path, "r", encoding="utf-8") as f:
            merges = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        writer.add_token_merges(merges)

    special_vocab = gguf.SpecialVocab(src_dir)
    special_vocab.add_to_gguf(writer)

    # Tensor Mapping
    safetensor_file = src_dir / "model.safetensors"
    if not safetensor_file.exists():
        raise FileNotFoundError(f"Missing model.safetensors in {src_dir}")

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
            continue
        gguf_name = tensor_map[hf_name]
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

    if "output.weight" not in [tensor_map.get(k) for k in weights.keys()] and token_embd_tensor is not None:
        writer.add_tensor("output.weight", token_embd_tensor, raw_dtype=gguf.GGMLQuantizationType.F16)
        added_tensors += 1

    print(f"[*] Writing {added_tensors} tensors to binary GGUF...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"[+] Successfully converted: {out_file} ({mb:.2f} MB)")
    return out_file


def convert_model(model_dir: str, out_file: str = "model.gguf"):
    """Entrypoint function to convert model to GGUF format."""
    src = Path(model_dir)
    dest = Path(out_file)
    try:
        return convert_with_native_gguf(src, dest)
    except Exception as e:
        print(f"[!] Native conversion failed: {e}. Trying llama.cpp fallback...")
        llama_cpp_script = Path("llama.cpp/convert_hf_to_gguf.py")
        if llama_cpp_script.exists():
            cmd = [sys.executable, str(llama_cpp_script), str(src), "--outfile", str(dest), "--outtype", "f16"]
            subprocess.check_call(cmd)
            return dest
        else:
            raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SafeTensors to GGUF")
    parser.add_argument("--model_dir", type=str, default="./medical_merged_model", help="Path to model directory")
    parser.add_argument("--output_file", type=str, default="./model.gguf", help="Output .gguf file path")
    args = parser.parse_args()

    convert_model(args.model_dir, args.output_file)
