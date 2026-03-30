"""
Step 9: GGUF Export & Quantization

Purpose:
    Merges local LoRA adapters into the base model and converts the 
    resulting standalone model into the GGUF format for use with llama.cpp.

Inputs:
    - TLF_CONFIG.json (For model paths and engine detection)
    - raw/models/ (Source of adapter weights)

Outputs:
    - raw/models/merged/ (Temporary full-weight FP16 model)
    - [ModelName].gguf (Quantized universal model file)

Usage:
    python code/dict/09_export_gguf.py [--quantization Q4_K_M] [--include-tokenizer]
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from utility import *

def main():
    parser = argparse.ArgumentParser(description="Export TLF Model to GGUF")
    parser.add_argument("--quantization", type=str, default="Q4_K_M", help="Quantization type (e.g., Q4_K_M, Q8_0, F16)")
    parser.add_argument("--out_dir", type=str, default="raw/models/GGUF/", help="Directory to save the GGUF file")
    args = parser.parse_args()

    # 1. Setup Logger
    logger = setup_logger("GGUF_Export", step_name="publishing")
    logger.info("Starting GGUF Export process...")

    # 2. Extract Configuration
    ft_config = TLF_CONFIG.get("finetuning", {})
    model_id = ft_config.get("model_id", "Unknown")
    engine = ft_config.get("engine", "hf")
    model_name = TLF_CONFIG.get("model_name", "TLF-7B-LLM-01")
    
    # 3. Determine Paths
    base_output_dir = ft_config.get("output_dir", "raw/models/")
    safe_model_id = model_id.replace('/', '_')
    target_name = f"{model_name}_{safe_model_id}"
    adapter_path = os.path.join(REPO_ROOT, base_output_dir, target_name)
    
    merged_path = os.path.join(REPO_ROOT, "raw/models/merged", target_name)
    gguf_out_dir = os.path.join(REPO_ROOT, args.out_dir)
    os.makedirs(gguf_out_dir, exist_ok=True)
    os.makedirs(merged_path, exist_ok=True)

    # 4. Handle MLX vs HF Merging
    if engine == "mlx":
        logger.info("Detected MLX training. Merging MLX adapters...")
        try:
            import mlx_lm
            from mlx_lm.utils import fetch_from_hub, load
            
            logger.info("Note: MLX natively supports GGUF via llama.cpp. Merging to full weights first...")
            # For MLX, we merge to a new model directory
            cmd = f"python3 -m mlx_lm.fuse --model {model_id} --adapter-path {adapter_path} --save-path {merged_path}"
            logger.info(f"Executing: {cmd}")
            subprocess.run(cmd.split(), check=True)
            
        except ImportError:
            logger.error("mlx_lm not found. Required for MLX merging.")
            return
        except Exception as e:
            logger.error(f"MLX fusing failed: {e}")
            return
    else:
        logger.info(f"Merging HF PEFT adapter from {adapter_path}...")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            logger.info(f"Loading base model: {model_id}")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float16,
                device_map="cpu" # CPU merge to avoid OOM
            )
            
            logger.info("Applying LoRA adapter...")
            model = PeftModel.from_pretrained(base_model, adapter_path)
            
            logger.info("Merging weights (Merge & Unload)...")
            merged_model = model.merge_and_unload()
            
            logger.info(f"Saving merged FP16 model to {merged_path}...")
            merged_model.save_pretrained(merged_path)
            
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.save_pretrained(merged_path)
            
        except Exception as e:
            logger.error(f"HF Merging failed: {e}")
            return

    # 5. GGUF Conversion Logic (Placeholder for llama.cpp integration)
    logger.info("--- GGUF CONVERSION ---")
    logger.info(f"Merged model is ready at: {merged_path}")
    logger.info("To complete GGUF conversion, you need llama.cpp installed.")
    
    conversion_cmd = f"python3 llamacpp/convert.py {merged_path} --outfile {gguf_out_dir}/{model_name}.gguf"
    quant_cmd = f"./llamacpp/quantize {gguf_out_dir}/{model_name}.gguf {gguf_out_dir}/{model_name}_{args.quantization}.gguf {args.quantization}"
    
    logger.info(f"Step A (Convert to GGUF): {conversion_cmd}")
    logger.info(f"Step B (Quantize): {quant_cmd}")

    print(f"\n[SUCCESS] Merged model prepared at {merged_path}")
    print(f"Follow the logged llama.cpp commands to finalize the {args.quantization} GGUF file.")

if __name__ == "__main__":
    main()
