import os
"""
TLF-7B-LLM-01: Master Pipeline Runner

Description:
    Orchestrates the end-to-end Supervised Fine-Tuning (SFT) lifecycle. 
    Sequentially executes database extraction, chunking, processing, 
    dataset building, fine-tuning, testing, and publishing.

Inputs:
    - tlf_config.json (Central control for all child scripts)
    - --step (Optional: 1, 2, 3, 4, 5, 6, 7, 8, or 'all')
    - --force-restart (Wipe outputs for Step 5)
    - --resume-from-checkpoint (Resume training for Step 5)

Outputs:
    - Standardized logs in logs/pipeline_master.log
    - Execution of all child scripts in code/dict/

Usage:
    python code/dict/run_pipeline.py --step all
"""
import sys
import subprocess
import glob
import json
import argparse
import argparse
from typing import List
from utility import *

config = load_config()

# Universal Logger Setup
logger = setup_logger("Pipeline-Master", step_name="pipeline_master")

def run_step(command_list: List[str], description: str) -> None:
    """
    Safely executes a shell command as a subprocess, logs its output 
    in real-time, and handles non-zero exit codes.
    
    Args:
        command_list (List[str]): The command and its arguments (e.g., ['python', 'script.py']).
        description (str): A human-readable description of the step for logging.

    Returns:
        None (Raises SystemExit on failure).
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Executing: {description}")
    logger.info(f"{'='*50}")
    
    result = subprocess.run(command_list, cwd=REPO_ROOT)
    if result.returncode != 0:
        logger.error(f"\n[ERROR] Step failed: {description}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="TLF-7B-LLM-01 Data Pipeline")
    parser.add_argument("--step", choices=["1", "2", "3", "4", "5", "6", "7", "8", "all"], default="all",
                        help="Which step to run. 1:Extract, 2:Chunk, 3:Process, 4:Build Dataset, 5:Finetune, 6:Test Inference, 7:Publish, 8:Analyze Stats")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Optional specific file to train on in Step 5 (or test individually)")
    parser.add_argument("--force-restart", action="store_true",
                        help="Wipes existing model directory during Step 5 to restart from scratch.")
    parser.add_argument("--resume-from-checkpoint", action="store_true",
                        help="Resumes MLX training from the last saved adapter weights.")
    parser.add_argument("--skip-iters", type=int, default=0,
                        help="Manually skip the first N iterations in Step 5.")
    parser.add_argument("--repo_id", type=str, default=None,
                        help="HuggingFace repository ID, required for Step 7 (Publishing).")
    args = parser.parse_args()

    logger.info(f"Starting TLF-7B-LLM-01 Data Pipeline (Step: {args.step})...")

    run_all = args.step == "all"
    
    if run_all or args.step == "1":
        run_step([sys.executable, os.path.join(SCRIPT_DIR, "01_extract_db.py")], "Step 1: Database Extraction")

    if run_all or args.step == "2":
        run_step([sys.executable, os.path.join(SCRIPT_DIR, "02_chunk_data.py")], "Step 2: File Chunking")

    if run_all or args.step == "3":
        logger.info(f"\n{'='*50}\nExecuting: Step 3: Chunk Processing\n{'='*50}")
        input_dir = os.path.join(REPO_ROOT, config['paths']['jsonl_input_dir'])
        output_dir = os.path.join(REPO_ROOT, config['paths']['jsonl_output_dir'])
        os.makedirs(output_dir, exist_ok=True)
        
        input_files = glob.glob(os.path.join(input_dir, "dictionary_part*.jsonl"))
        if not input_files:
            logger.info("[WARNING] No chunk files found to process.")
            
        for in_file in sorted(input_files):
            filename = os.path.basename(in_file)
            part_num = filename.replace('dictionary_part', '').replace('.jsonl', '')
            out_file = os.path.join(output_dir, f"dictionary_processed_{part_num}.jsonl")
            
            if os.path.exists(out_file):
                logger.info(f"Skipping {filename} -> Output already exists.")
                continue
                
            cmd = [
                sys.executable,
                os.path.join(SCRIPT_DIR, "03_process_chunk.py"),
                "--input_file", in_file,
                "--output_file", out_file
            ]
            result = subprocess.run(cmd, cwd=REPO_ROOT)
            if result.returncode != 0:
                logger.info(f"\n[ERROR] Step 3 failed on {filename}")
                sys.exit(result.returncode)

    if run_all or args.step == "4":
        run_step([sys.executable, os.path.join(SCRIPT_DIR, "04_build_dataset.py")], "Step 4: SFT Dataset Generation")
        
    if run_all or args.step == "5":
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "05_finetune_model.py")]
        if args.input_file:
            cmd.extend(["--input_file", args.input_file])
        if args.force_restart:
            cmd.append("--force-restart")
        if args.resume_from_checkpoint:
            cmd.append("--resume-from-checkpoint")
        if args.skip_iters:
            cmd.extend(["--skip-iters", str(args.skip_iters)])
        run_step(cmd, "Step 5: LLaMA-2 QLoRA Fine-tuning")
        
    if run_all or args.step == "6":
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "06_test_model.py")]
        run_step(cmd, "Step 6: LoRA Adapter Interactive CLI Inference")
        
    if run_all or args.step == "7":
        if not args.repo_id:
            logger.info("\n[ERROR] --repo_id is required to run Step 7 (Publishing).")
            logger.info("Example: python run_pipeline.py --step 7 --repo_id your-username/TLF-7B-LLM-01")
            sys.exit(1)
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "07_publish_model.py"), "--repo_id", args.repo_id]
        run_step(cmd, f"Step 7: Publishing Adapter to HuggingFace ({args.repo_id})")

    if run_all or args.step == "8":
        run_step([sys.executable, os.path.join(SCRIPT_DIR, "08_analyze_training.py")], "Step 8: Training Analytics & Plotting")
    
    logger.info(f"{'*'*50}")
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"{'*'*50}\n")

if __name__ == "__main__":
    main()
