"""
Step 8: Training Analytics & Loss Plotting

Purpose:
    Visualizes the training progress by parsing loss statistics from MLX 
    or HuggingFace CSV logs. Generates a combined performance plot.

Inputs:
    - logs/mlx/training_stats.csv
    - logs/hf/training_stats.csv

Outputs:
    - logs/training_progress.png (The visualization)
    - Consolidated statistics printed to console.

Usage:
    python code/dict/08_analyze_training.py
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import json
from typing import Optional
from utility import * 

logs_dir = LOGS_DIR

# Universal Logger Setup
logger = setup_logger("Analysis", step_name="analysis")

def analyze_engine(engine_name: str) -> Optional[pd.DataFrame]:
    """
    Parses the CSV log for a specific engine (MLX/HF) and returns 
    a cleaned DataFrame ready for plotting.
    
    Args:
        engine_name (str): The name of the engine to analyze ('mlx' or 'hf').

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing the parsed training stats,
            or None if the log file is missing or empty.
    """
    engine_logs = os.path.join(LOGS_DIR, engine_name)
    stats_file = os.path.join(engine_logs, "training_stats.csv")
    
    if not os.path.exists(stats_file):
        logger.info(f"[INFO] No training stats found for {engine_name.upper()}.")
        return None

    try:
        df = pd.read_csv(stats_file)
        if df.empty:
            logger.warning(f"[WARNING] Stats file for {engine_name.upper()} is empty.")
            return None

        logger.info(f"\n--- {engine_name.upper()} Training Summary ---")
        logger.info(f"Total Iterations/Steps: {len(df)}")
        
        # Loss Analysis
        if 'TrainLoss' in df.columns: # MLX naming
            t_col, v_col = 'TrainLoss', 'ValLoss'
        else: # HF naming
            t_col, v_col = 'loss', 'eval_loss'
            
        if t_col in df.columns:
            logger.info(f"Initial Train Loss: {df[t_col].iloc[0]:.4f}")
            logger.info(f"Final Train Loss:   {df[t_col].iloc[-1]:.4f}")
            logger.info(f"Minimum Train Loss: {df[t_col].min():.4f}")

        if v_col in df.columns and not df[v_col].dropna().empty:
            logger.info(f"Final Val Loss:     {df[v_col].dropna().iloc[-1]:.4f}")

        return df
    except Exception as e:
        logger.error(f"[ERROR] Failed to analyze {engine_name}: {e}")
        return None

def plot_combined(mlx_df, hf_df):
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    has_data = False
    
    if mlx_df is not None:
        plt.plot(mlx_df['Iter'], mlx_df['TrainLoss'], label='MLX Train Loss', color='blue', alpha=0.7)
        if 'ValLoss' in mlx_df.columns:
            val_df = mlx_df.dropna(subset=['ValLoss'])
            if not val_df.empty:
                plt.plot(val_df['Iter'], val_df['ValLoss'], 'o--', label='MLX Val Loss', color='cyan')
        has_data = True

    if hf_df is not None:
        # Step/Epoch mapping might differ, but we plot by Step for now
        plt.plot(hf_df['step'], hf_df['loss'], label='HF Train Loss', color='red', alpha=0.7)
        has_data = True

    if not has_data:
        return

    plt.title("Training Loss Progression (TLF-7B-LLM-01)")
    plt.xlabel("Iteration / Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.yscale('log') # Log scale helps see convergence detail
    
    plot_path = os.path.join(logs_dir, "training_progress.png")
    plt.savefig(plot_path)
    logger.info(f"\n[SUCCESS] Progress plot saved to: {plot_path}")

def main():
    logger.info("Starting Training Analytics...")
    
    mlx_stats = analyze_engine("mlx")
    hf_stats = analyze_engine("hf")
    
    if mlx_stats is not None or hf_stats is not None:
        plot_combined(mlx_stats, hf_stats)
    else:
        logger.info("\n[SKIP] No data available to plot.")

if __name__ == "__main__":
    main()
