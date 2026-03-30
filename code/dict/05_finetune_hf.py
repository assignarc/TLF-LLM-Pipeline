"""
Step 5: HuggingFace Fine-tuning (Nvidia/PyTorch Optimization)

Purpose:
    Executes LLaMA-2 4-bit QLoRA training using the HuggingFace Transformers 
    and PEFT libraries. Optimized for Nvidia GPUs via CUDA.

Inputs:
    - raw/dict/4-prompts/ (Directory containing training instructions)
    - TLF_CONFIG.json [finetuning] (Hyperparameters)

Outputs:
    - raw/models/llama2-dict-finetuned/ (Standard PEFT adapter checkpoints)
    - logs/hf/training_stats.csv (Epoch-by-epoch loss tracking)

Usage:
    python code/dict/05_finetune_hf.py [--input_file <path>] [--force-restart] [--resume-from-checkpoint] [--iters <n>] [--max-seq-length <n>] [--max-nan-threshold <n>]
"""

import os
import argparse
import json
import torch
import logging
import shutil
import sys
from utility import *

TARGET_MODEL_NAME = TLF_CONFIG.get("model_name", 'TLF-7B-HF-01')

def main() -> None:
    """
    Initializes HF training, sets bitsandbytes config, and invokes the 
    SFTTrainer. Logs stats to CSV for Step 8.
    
    Args:
        None (Uses argparse for input).

    Returns:
        None (Writes checkpoints to raw/models/ and stats to logs/hf/).
    """
    parser = argparse.ArgumentParser(description="HuggingFace LLaMA-2 Fine-tuning")
    parser.add_argument("--input_file", type=str, help="Specific JSONL file to train on (absolute or relative path)", default=None)
    parser.add_argument("--force-restart", action="store_true", help="Delete existing model output directory and start from scratch")
    parser.add_argument("--resume-from-checkpoint", action="store_true", help="Resume training from previous checkpoints")
    parser.add_argument("--iters", type=int, help="Override number of iterations (max_steps)", default=None)
    parser.add_argument("--max-seq-length", type=int, help="Override max sequence length", default=None)
    parser.add_argument("--skip-iters", type=int, default=0, help="Manually skip the first N iterations")
    parser.add_argument("--max-nan-threshold", type=int, help="Override max consecutive NaN threshold", default=None)
    parser.add_argument("--nan-action", type=str, choices=["terminate", "rollback", "rollback-skip"], help="Action on reaching NaN threshold", default=None)
    args = parser.parse_args()

        
    ft_config = TLF_CONFIG.get("finetuning", {})
    model_id = ft_config.get("model_id", "meta-llama/Llama-2-7b-hf")
    hf_token = ft_config.get("hf_token", None)
    if hf_token == "your_huggingface_token_here": hf_token = None
    
    if hf_token:
        masked_token = hf_token[:4] + "*" * (len(hf_token) - 8) + hf_token[-4:]
        print(f"Using HuggingFace Token: {masked_token}")
    else:
        print("No HuggingFace Token detected in tlf_config.json. HF will use local cache or fallback auth.")
        
    safe_model_id = model_id.replace('/', '_')
    target_name = f"{TARGET_MODEL_NAME}_{safe_model_id}"
    base_output_dir = ft_config.get("output_dir", "raw/models/")
    output_dir = os.path.join(REPO_ROOT, base_output_dir, target_name)

    epochs = ft_config.get("epochs", 3)
    batch_size = ft_config.get("batch_size", 4)
    lr = ft_config.get("learning_rate", 2e-4)
    lora_r = ft_config.get("lora_r", 64)
    lora_alpha = ft_config.get("lora_alpha", 16)
    lora_dropout = ft_config.get("lora_dropout", 0.1)
    max_seq_length = args.max_seq_length or ft_config.get("max_seq_length", 1024)
    grad_accum_steps = ft_config.get("gradient_accumulation_steps", 4)
    optim_type = ft_config.get("optim", "paged_adamw_32bit")
    save_steps = ft_config.get("save_steps", 100)
    log_steps = ft_config.get("logging_steps", 10)
    max_grad_norm = ft_config.get("max_grad_norm", 0.3)
    warmup_ratio = ft_config.get("warmup_ratio", 0.03)
    lr_scheduler = ft_config.get("lr_scheduler_type", "constant")

    if args.force_restart and os.path.exists(output_dir):
        print(f"Force restarting. Deleting existing model directory: {output_dir}")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    
    # Universal Logger Setup
    log_folder = os.path.join(LOGS_DIR, "hf", safe_model_id)
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, "training.log")
    logger = setup_logger("HF-Finetune", log_file)
    
    logger.info(f"Starting HF Fine-tuning for {model_id}")

    prompts_dir = os.path.join(REPO_ROOT, config['paths']['prompts_output_dir'])

    if args.input_file:
        data_files = [args.input_file]
        logger.info(f"Loading single dataset file: {args.input_file}")
    else:
        os.makedirs(prompts_dir, exist_ok=True)
        data_files = glob.glob(os.path.join(prompts_dir, "instruction_dataset_*.jsonl"))
        if not data_files:
            logger.error(f"No processed prompt files found in {prompts_dir}")
            return
        logger.info(f"Found {len(data_files)} instruction dataset chunks.")

    # Stats Logging Setup
    safe_model_id = model_id.replace('/', '_') if model_id else "unknown_model"
    log_base = os.path.join(REPO_ROOT, "logs", "hf", safe_model_id)
    os.makedirs(log_base, exist_ok=True)
    stats_file = os.path.join(log_base, "training_stats.csv")

    # Log Versioning/Rotation
    if args.force_restart and os.path.exists(stats_file):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_file = os.path.join(log_base, f"training_stats_{timestamp}.csv")
        os.rename(stats_file, versioned_file)
        logger.info(f"Archived previous logs to {versioned_file}")

    file_exists = os.path.exists(stats_file)
    write_mode = 'a' if (args.resume_from_checkpoint and file_exists) else 'w'

    logger.info("Attempting to load standard ML libraries (datasets, torch, transformers, peft, trl)...")
    try:
        from datasets import load_dataset
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
            TrainerCallback
        )
        import transformers
        import huggingface_hub
        from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
        from trl import SFTTrainer, SFTConfig
        import csv
        from datetime import datetime

        class CSVLoggerCallback(TrainerCallback):
            def __init__(self, csv_path, mode='w'):
                self.csv_path = csv_path
                self.mode = mode
                self.header_written = (mode == 'a' and os.path.exists(csv_path))

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    metrics = {k: v for k, v in logs.items() if k in ['loss', 'learning_rate', 'epoch', 'step', 'grad_norm']}
                    if not metrics: return
                    metrics['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(self.csv_path, 'a' if self.header_written else self.mode, newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=metrics.keys())
                        if not self.header_written:
                            writer.writeheader()
                            self.header_written = True
                        writer.writerow(metrics)

        class NaNDetectionCallback(TrainerCallback):
            def __init__(self, threshold=1):
                self.threshold = threshold
                self.consecutive_nans = 0

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and 'loss' in logs:
                    val = logs['loss']
                    if str(val).lower() == 'nan' or (isinstance(val, float) and val != val):
                        self.consecutive_nans += 1
                        logger.warning(f"NaN loss detected in HF at Step {state.global_step}. (Consecutive: {self.consecutive_nans}/{self.threshold})")
                        
                        if self.consecutive_nans >= self.threshold:
                            nan_action = args.nan_action or ft_config.get("nan_action", "rollback")
                            logger.critical(f"NaN threshold ({self.threshold}) reached in HF! Action: {nan_action.upper()}")
                            
                            # Diagnostic Capture
                            it_num = state.global_step
                            
                            # Attempt to pull the actual record for logging
                            try:
                                real_idx = (it_num - 1) * grad_accum_steps * batch_size
                                record = dataset["train"][real_idx]
                            except:
                                record = {"text": "Unknown (HF Data Access Error)"}

                            debug_file = save_nan_diagnostic(it_num, record, note=f"NaN loss detected in HF at Step {it_num}")
                            logger.info(f"[DIAGNOSTIC] Metadata saved to {debug_file}")

                            if "skip" in nan_action:
                                log_skipped_record(it_num, record, output_dir)
                                logger.warning(f"AUTO-SKIP: Iteration {it_num} added to HF blacklist.")
                            
                            raise Exception("NaN_CASCADE")
                        else:
                            logger.info(f"Continuing HF training as per max_nan_threshold={self.threshold}")
                    else:
                        # Reset counter on valid loss
                        self.consecutive_nans = 0
        
        logging.getLogger("huggingface_hub").setLevel(logging.DEBUG)
        logging.getLogger("transformers").setLevel(logging.INFO)
        logging.getLogger("urllib3").setLevel(logging.DEBUG)
        
    except ImportError as e:
        logger.error(f"Missing python library: {e}")
        return

    # Load and Filter Dataset
    logger.info("Loading instruction mappings...")
    full_dataset = load_dataset("json", data_files={"train": data_files})["train"]
    
    manual_skip_count = args.skip_iters * batch_size * grad_accum_steps

    device = ft_config.get("device", "auto")
    try:
        if device == "mps":
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="mps", token=hf_token)
        else:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map=device, token=hf_token)
            
        model.config.use_cache = False
        if device == "mps" and "paged" in optim_type:
            optim_type = "adamw_torch"
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        if device != "mps":
            model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return

    total_steps = len(dataset["train"]) // (batch_size * grad_accum_steps) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        optim=optim_type,
        save_steps=save_steps,
        save_total_limit=5,
        logging_steps=log_steps,
        learning_rate=lr,
        max_grad_norm=max_grad_norm,
        num_train_epochs=epochs if not args.iters else None,
        max_steps=args.iters if args.iters else -1,
        warmup_steps=warmup_steps,
        lr_scheduler_type=lr_scheduler,
        dataset_text_field="text",
        max_length=max_seq_length,
    )

    MAX_ROLLBACKS = ft_config.get("max_rollbacks", 25)
    rollback_count = 0
    nan_action = ft_config.get("nan_action", "rollback")

    while rollback_count <= MAX_ROLLBACKS:
        try:
            # 1. Dynamically Filter Dataset for Current Rollback State
            blacklist = get_nan_blacklist(output_dir)
            blacklisted_indices = set()
            for bad_it in blacklist:
                start = (int(bad_it) - 1) * grad_accum_steps * batch_size
                for i in range(start, start + grad_accum_steps * batch_size):
                    blacklisted_indices.add(i)

            valid_indices = [i for i in range(len(full_dataset)) 
                             if i >= manual_skip_count and i not in blacklisted_indices]
            
            dataset = full_dataset.select(valid_indices)
            logger.info(f"HF Dataset ready: {len(dataset)} records (Skipped: {len(full_dataset) - len(dataset)}).")

            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                peft_config=peft_config,
                processing_class=tokenizer,
                args=sft_config,
                callbacks=[
                    CSVLoggerCallback(stats_file, mode=write_mode), 
                    NaNDetectionCallback(threshold=ft_config.get("max_nan_threshold", 1))
                ]
            )
            
            # Checkpoint Resumption Logic
            resume_from_checkpoint = False
            if args.resume_from_checkpoint:
                resume_from_checkpoint = True
            else:
                checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
                if checkpoints and not args.force_restart:
                    resume_from_checkpoint = True
                    logger.info(f"Found existing checkpoints in {output_dir}. Auto-resuming...")

            logger.info(f"Initiating SFTTrainer Training Loop (Attempt {rollback_count + 1})...")
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            trainer.model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Fine-tuning complete. LoRA adapter and tokenizer saved to {output_dir}")
            break # Exit the rollback loop on success
            
        except Exception as e:
            if "NaN_CASCADE" in str(e):
                rollback_count += 1
                if "rollback" in nan_action and rollback_count <= MAX_ROLLBACKS:
                    logger.warning(f"NaN cascade intercepted. Triggering HF rollback {rollback_count}/{MAX_ROLLBACKS}...")
                    continue
                else:
                    logger.error(f"NaN action '{nan_action}' or max retries ({MAX_ROLLBACKS}) reached. Terminating.")
                    break
            else:
                logger.error(f"Training failed midway: {e}")
                break

if __name__ == "__main__":
    main()
