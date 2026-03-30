"""
Step 5: LLaMA-2 Fine-tuning Dispatcher

Purpose:
    Acts as the entry point for the fine-tuning stage. Reads tlf_config.json to 
    determine the target engine (MLX for Apple Silicon or HuggingFace for Nvidia) 
    and forwards CLI arguments to the respective specialized script.

Inputs:
    - tlf_config.json [finetuning][engine]
    - --input_file (Optional path to a single JSONL training file)
    - --force-restart (Boolean to wipe existing output directories)
    - --resume-from-checkpoint (Boolean to resume from last saved state)
    - --iters (Optional override for total training iterations)
    - --max-seq-length (Optional override for context length)
    - --max-nan-threshold (Optional override for consecutive NaN tolerance)

Outputs:
    - Invokes 05_finetune_mlx.py or 05_finetune_hf.py

Usage:
    python code/dict/05_finetune_model.py [--input_file <path>] [--force-restart] [--resume-from-checkpoint]
"""

import os
import json
import argparse
import subprocess
import sys
from utility import * 

def main() -> None:
    """
    Parses CLI arguments, selects the fine-tuning script, and executes it 
    as a subprocess to ensure clean environment separation.
    
    Args:
        None (Uses argparse for input).

    Returns:
        None (Invokes subprocess).
    """
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA-2 Dispatcher (HF vs MLX)")
    parser.add_argument("--input_file", type=str, help="Specific JSONL file to train on", default=None)
    parser.add_argument("--force-restart", action="store_true", help="Delete existing model output directory and start from scratch")
    parser.add_argument("--resume-from-checkpoint", action="store_true", help="Resume training from previous weights")
    parser.add_argument("--iters", type=int, help="Override number of iterations", default=None)
    parser.add_argument("--max-seq-length", type=int, help="Override max sequence length", default=None)
    parser.add_argument("--skip-iters", type=int, default=0, help="Manually skip the first N iterations")
    parser.add_argument("--max-nan-threshold", type=int, help="Override max consecutive NaN threshold", default=None)
    parser.add_argument("--nan-action", type=str, choices=["terminate", "rollback", "rollback-skip"], help="Action on reaching NaN threshold", default=None)
    args = parser.parse_args()
        
    engine = TLF_CONFIG.get("finetuning", {}).get("engine", "hf")
    
    # Universal Logger Setup
    logger = setup_logger("Finetune-Dispatcher", step_name="finetune_dispatch")

    logger.info(f"Fine-tuning Engine Selected: {engine.upper()}")
    
    if engine == "mlx":
        target_script = os.path.join(SCRIPT_DIR, "05_finetune_mlx.py")
    else:
        target_script = os.path.join(SCRIPT_DIR, "05_finetune_hf.py")
        
    cmd = [sys.executable, target_script]
    if args.input_file:
        cmd.extend(["--input_file", args.input_file])
    if args.force_restart:
        cmd.append("--force-restart")
    if args.resume_from_checkpoint:
        cmd.append("--resume-from-checkpoint")
    if args.iters:
        cmd.extend(["--iters", str(args.iters)])
    if args.max_seq_length:
        cmd.extend(["--max-seq-length", str(args.max_seq_length)])
    if args.skip_iters:
        cmd.extend(["--skip-iters", str(args.skip_iters)])
    if args.max_nan_threshold:
        cmd.extend(["--max-nan-threshold", str(args.max_nan_threshold)])
    if args.nan_action:
        cmd.extend(["--nan-action", args.nan_action])
        
    logger.info(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
