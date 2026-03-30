"""
Step 5: MLX LLaMA-2 Fine-tuning (Apple Silicon Optimization)

Purpose:
    Executes LLaMA-2 QLoRA training using the MLX framework, optimized 
    for Apple Silicon GPUs (Metal).

Inputs:
    - raw/dict/4-prompts/ (Directory containing training instructions)
    - TLF_CONFIG.json [finetuning] (Hyperparameters)

Outputs:
    - raw/models/TLF-7B-MLX-01/adapters.safetensors (MLX adapter weights)
    - logs/mlx/training_stats.csv (Iteration-by-iteration loss tracking)

Usage:
    python code/dict/05_finetune_mlx.py [--input_file <path>] [--force-restart] [--resume-from-checkpoint] [--iters <n>] [--max-seq-length <n>] [--max-nan-threshold <n>]
"""

import sys
import os
import json
import subprocess
import argparse
import glob
import shutil
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from utility import * 

# Centralized save_nan_diagnostic is imported from utility

TARGET_MODEL_NAME = TLF_CONFIG.get("model_name", 'TLF-7B-MLX-01')
MAX_ROLLBACKS = 100

def rotate_checkpoints(directory: str, keep_last: int = 5, logger=None) -> None:
    """
    Scans the directory for *_adapters.safetensors files and deletes 
    all but the last N, sorted by iteration number.
    """
    import re
    pattern = os.path.join(directory, "*_adapters.safetensors")
    checkpoints = glob.glob(pattern)
    
    # We only care about numbered checkpoints like 0010000_adapters.safetensors
    # The 'adapters.safetensors' file should be excluded from rotation logic
    numbered_checkpoints = []
    for cp in checkpoints:
        base = os.path.basename(cp)
        if base == "adapters.safetensors":
            continue
        numbered_checkpoints.append(cp)
        
    if len(numbered_checkpoints) <= keep_last:
        return
        
    def get_iteration(path):
        base = os.path.basename(path)
        # Assuming format like 0010000_adapters.safetensors
        match = re.search(r"(\d+)_adapters", base)
        return int(match.group(1)) if match else 0
        
    numbered_checkpoints.sort(key=get_iteration)
    
    to_delete = numbered_checkpoints[:-keep_last]
    for cp in to_delete:
        try:
            os.remove(cp)
            if logger:
                logger.info(f"Cleanup: Deleted old checkpoint {os.path.basename(cp)} to save space.")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to delete {cp}: {e}")

def find_last_stable_checkpoint(directory: str) -> Optional[str]:
    """
    Identifies the highest-numbered adapter checkpoint in the directory.
    """
    import re
    pattern = os.path.join(directory, "*_adapters.safetensors")
    checkpoints = glob.glob(pattern)
    
    stable_checkpoints = []
    for cp in checkpoints:
        base = os.path.basename(cp)
        if base == "adapters.safetensors":
            continue
        match = re.search(r"(\d+)_adapters", base)
        if match:
            stable_checkpoints.append((int(match.group(1)), cp))
            
    if not stable_checkpoints:
        return None
        
    stable_checkpoints.sort(key=lambda x: x[0], reverse=True)
    return stable_checkpoints[0][1]


def main() -> None:
    """
    Initializes MLX training, sets hyperparameters from tlf_config.json, 
    and invokes the MLX-LM LoRA trainer. Logs stats to CSV for Step 8.
    """
    parser = argparse.ArgumentParser(description="MLX LLaMA-2 Fine-tuning")
    parser.add_argument("--input_file", type=str, help="Specific instruction file to train on")
    parser.add_argument("--force-restart", action="store_true", help="Delete existing model first")
    parser.add_argument("--resume-from-checkpoint", action="store_true", help="Resume from last adapter")
    parser.add_argument("--iters", type=int, help="Override number of iterations")
    parser.add_argument("--max-seq-length", type=int, help="Override max sequence length")
    parser.add_argument("--skip-iters", type=int, default=0, help="Manually skip the first N iterations")
    parser.add_argument("--max-nan-threshold", type=int, help="Override max consecutive NaN threshold")
    parser.add_argument("--nan-action", type=str, choices=["terminate", "rollback", "rollback-skip"], help="Action on reaching NaN threshold")
    args = parser.parse_args()
        
    ft_config = TLF_CONFIG.get("finetuning", {})
    model_id = ft_config.get("model_id", "meta-llama/Llama-2-7b-hf")
    safe_model_id = model_id.replace('/', '_')
    target_name = f"{TARGET_MODEL_NAME}_{safe_model_id}"
    
    base_output_dir = ft_config.get("output_dir", "raw/models/")
    output_dir = os.path.join(REPO_ROOT, base_output_dir, target_name)
    MAX_ROLLBACKS = ft_config.get("max_rollbacks", 100)
    # Universal Logger Setup
    logger = setup_logger("MLX-Finetune", step_name="mlx_training")
    logger.info(f"Starting MLX Fine-tuning for {model_id}")
    
    # MLX specific params
    epochs = ft_config.get("epochs", 3)
    batch_size = ft_config.get("batch_size", 4)
    lr = ft_config.get("learning_rate", 2e-4)
    lora_r = ft_config.get("lora_r", 64)
    lora_alpha = ft_config.get("lora_alpha", 16)
    max_nan_threshold = args.max_nan_threshold or ft_config.get("max_nan_threshold", 1)
    max_seq_length = args.max_seq_length or ft_config.get("max_seq_length", 1024)
    save_steps = ft_config.get("save_steps", 100)
    grad_accumulation_steps = ft_config.get("gradient_accumulation_steps", 1)
    
    if args.force_restart and os.path.exists(output_dir):
        logger.info(f"Force restarting. Deleting existing MLX model directory: {output_dir}")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for MLX-LM
    mlx_data_dir = os.path.join(output_dir, "data")
    os.makedirs(mlx_data_dir, exist_ok=True)
    prompts_dir = os.path.join(REPO_ROOT, TLF_CONFIG['paths']['prompts_output_dir'])
    
    if args.input_file:
        source_files = [args.input_file]
        logger.info(f"Using single input file: {os.path.basename(args.input_file)}")
    else:
        source_files = sorted(glob.glob(os.path.join(prompts_dir, "instruction_dataset_*.jsonl")))
        if not source_files:
            logger.error(f"No processed prompt files found in {prompts_dir}")
            return
        logger.info(f"Detected {len(source_files)} chunks. Combining for training...")

    # Create combined train.jsonl with filtering
    train_dest = os.path.join(mlx_data_dir, "train.jsonl")
    valid_dest = os.path.join(mlx_data_dir, "valid.jsonl")
    
    manual_skip_count = args.skip_iters * batch_size * grad_accumulation_steps

    if not os.path.exists(valid_dest) or args.force_restart:
        with open(source_files[0], 'r') as f_in, open(valid_dest, 'w') as f_out:
            for i, line in enumerate(f_in):
                if i >= 100: break
                f_out.write(line)

    logger.info(f"Data prepared in {mlx_data_dir}")

    # Check for initial adapter
    adapter_file = os.path.join(output_dir, "adapters.safetensors")
    
    # Calculate target iterations (placeholder until dataset is constructed)
    iters = args.iters if args.iters else 0
    logger.info(f"Target iterations set dynamically in training loop.")

    # Setup Stats File
    safe_model_id = model_id.replace('/', '_') if model_id else "unknown_model"
    log_base = os.path.join(REPO_ROOT, "logs", "mlx", safe_model_id)
    os.makedirs(log_base, exist_ok=True)
    stats_file = os.path.join(log_base, "training_stats.csv")

    if args.force_restart and os.path.exists(stats_file):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_file = os.path.join(log_base, f"training_stats_{timestamp}.csv")
        os.rename(stats_file, versioned_file)

    write_mode = 'a' if (args.resume_from_checkpoint and os.path.exists(stats_file)) else 'w'
    
    # Initial MLX Config
    mlx_config = {
        "model": model_id,
        "train": True,
        "data": mlx_data_dir,
        "iters": iters,
        "batch_size": batch_size,
        "learning_rate": lr,
        "grad_accumulation_steps": grad_accumulation_steps,
        "grad_checkpoint": True,
        "max_seq_length": max_seq_length,
        "max_grad_norm": ft_config.get("max_grad_norm", 0.1),
        "save_every": save_steps,
        "adapter_path": output_dir,
        "lora_parameters": {
            "rank": lora_r,
            "scale": lora_alpha,
            "dropout": ft_config.get("lora_dropout", 0.0),
        }
    }
    
    if args.resume_from_checkpoint or (os.path.exists(adapter_file) and not args.force_restart):
        mlx_config["resume_adapter_file"] = adapter_file

    config_path = os.path.join(output_dir, "mlx_config.yaml")
    import yaml
    import re
    import csv

    def sanitize_config_for_export(cfg: dict, base_dir: str) -> dict:
        """
        Return a copy of cfg with all absolute paths converted to paths
        relative to base_dir (the model directory). This prevents local
        filesystem paths from appearing in the uploaded mlx_config.yaml.
        """
        relative = dict(cfg)
        for key in ("adapter_path", "data", "resume_adapter_file"):
            if key in relative and relative[key]:
                try:
                    rel = os.path.relpath(relative[key], base_dir)
                    relative[key] = rel
                except ValueError:
                    pass  # Windows cross-drive edge-case; leave as-is
        return relative

    # Regex patterns
    stats_re = re.compile(r"Iter\s+(\d+):\s+(Train|Val)\s+loss\s+([nan\d\.]+)", re.IGNORECASE)
    metrics_re = re.compile(r"Learning Rate\s+([\d\.e-]+),\s+It/sec\s+([\d\.]+),\s+Tokens/sec\s+([\d\.]+),\s+Trained Tokens\s+(\d+),\s+Peak mem\s+([\d\.]+)")
    save_re = re.compile(r"Saved adapter weights to (.*) and (.*)\.")

    rollback_count = 0

    consecutive_nans = 0
    rollback_triggered = False

    try:
        with open(stats_file, write_mode, newline='') as csvfile:
            fieldnames = ['Iter', 'TrainLoss', 'ValLoss', 'LearningRate', 'ItSec', 'TokensSec', 'TrainedTokens', 'PeakMemGB', 'Timestamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_mode == 'w':
                writer.writeheader()

            while rollback_count <= MAX_ROLLBACKS:
                # 1. Dynamically Filter Dataset for Current Rollback State
                blacklist = get_nan_blacklist(output_dir)
                blacklisted_indices = set()
                for bad_it in blacklist:
                    start = (int(bad_it) - 1) * grad_accumulation_steps * batch_size
                    for i in range(start, start + grad_accumulation_steps * batch_size):
                        blacklisted_indices.add(i)
                
                logger.info(f"Filtering MLX dataset: Manual Skip={args.skip_iters} iters, Blacklist={len(blacklist)} iters.")
                global_idx, skipped_total = 0, 0
                with open(train_dest, 'w', encoding='utf-8') as outfile:
                    for fpath in source_files:
                        with open(fpath, 'r', encoding='utf-8') as infile:
                            for line in infile:
                                if (global_idx < manual_skip_count) or (global_idx in blacklisted_indices):
                                    skipped_total += 1
                                else:
                                    outfile.write(line)
                                global_idx += 1
                logger.info(f"Training set ready: {global_idx - skipped_total} records (Skipped: {skipped_total}).")

                # Dynamic Recalculation of target iterations
                if not args.iters:
                    num_examples = global_idx - skipped_total
                    iters = (num_examples // (batch_size * grad_accumulation_steps)) * epochs
                mlx_config["iters"] = iters

                # 2. Re-write config (in case of dynamic changes)
                with open(config_path, 'w') as f:
                    yaml.dump(sanitize_config_for_export(mlx_config, output_dir), f)
                cmd = [sys.executable, "-m", "mlx_lm.lora", "--config", config_path]
                logger.info(f"Executing: {' '.join(cmd)}")
                
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                current_stats = {"Iter": None}
                rollback_triggered = False
                
                for line in process.stdout:
                    clean_line = line.strip()
                    if not clean_line: continue
                    
                    if "%|" in clean_line and ("it/s" in clean_line or "s/it" in clean_line):
                        sys.stdout.write(f"\r{clean_line}")
                        sys.stdout.flush()
                    else:
                        logger.info(clean_line)

                    # Parse output
                    match = stats_re.search(clean_line)
                    if match:
                        it_num = match.group(1)
                        mode = match.group(2)
                        loss = match.group(3)
                        
                        if str(loss).lower() == "nan":
                            consecutive_nans += 1
                            logger.warning(f"NaN detected at Iteration {it_num}. ({consecutive_nans}/{max_nan_threshold})")
                            logger.warning(f"Clean Line: {clean_line}")
                            logger.warning(f"Rollback: {rollback_count}/{MAX_ROLLBACKS}")
                            if consecutive_nans >= max_nan_threshold:
                                nan_action = args.nan_action or ft_config.get("nan_action", "rollback")
                                logger.critical(f"Threshold reached. Action: {nan_action.upper()}")
                                
                                # Diagnostic
                                try:
                                    it_val = int(it_num)
                                    real_record_idx = (it_val - 1) * grad_accumulation_steps * batch_size
                                    with open(train_dest, 'r') as f_diag:
                                        for _ in range(real_record_idx): next(f_diag)
                                        record = json.loads(next(f_diag))
                                except: record = {"text": "Unknown"}
                                
                                debug_file = save_nan_diagnostic(it_num, record, note=f"NaN limit hit. Action: {nan_action}")
                                logger.info(f"Diagnostic saved: {debug_file}")
                                
                                if "rollback" in nan_action:
                                    if "skip" in nan_action:
                                        log_skipped_record(it_num, record, output_dir)
                                        logger.warning(f"AUTO-SKIP: Iteration {it_num} added to blacklist.")
                                    
                                    process.terminate()
                                    rollback_count += 1
                                    last_checkpoint = find_last_stable_checkpoint(output_dir)
                                    if last_checkpoint and rollback_count <= MAX_ROLLBACKS:
                                        logger.info(f"Rolling back to {os.path.basename(last_checkpoint)}")
                                        mlx_config["resume_adapter_file"] = last_checkpoint
                                        consecutive_nans = 0
                                        rollback_triggered = True
                                        break # Breaks for loop to restart Popen
                                    else:
                                        logger.error("No checkpoint or too many retries. Stopping.")
                                        return
                                else:
                                    process.terminate()
                                    return
                        else:
                            consecutive_nans = 0

                        # Store stats
                        if it_num != current_stats["Iter"] and current_stats["Iter"] is not None:
                            writer.writerow(current_stats)
                            csvfile.flush()
                        
                        if it_num != current_stats["Iter"]:
                            current_stats = {"Iter": it_num, "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                        current_stats[f"{mode}Loss"] = loss

                    # Metrics and Savign
                    if metrics_re.search(clean_line):
                        met = metrics_re.search(clean_line)
                        current_stats.update({
                            "LearningRate": met.group(1), "ItSec": met.group(2),
                            "TokensSec": met.group(3), "TrainedTokens": met.group(4),
                            "PeakMemGB": met.group(5)
                        })
                    
                    if save_re.search(clean_line):
                        rotate_checkpoints(output_dir, keep_last=5, logger=logger)

                process.wait()
                if rollback_triggered:
                    logger.info("Restarting from stable checkpoint...")
                    continue

                if process.returncode == 0 or consecutive_nans < max_nan_threshold:
                    if current_stats["Iter"]: writer.writerow(current_stats)
                    logger.info("Training finished successfully.")
                    return
                
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
