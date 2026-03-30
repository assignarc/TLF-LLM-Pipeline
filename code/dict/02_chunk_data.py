"""
Step 2: File Chunking (LMM Memory Optimization)

Purpose:
    Divides the massive dictionary JSONL file into smaller, fixed-size chunks to 
    prevent Out-Of-Memory (OOM) errors during subsequent ETL and training stages.

Inputs:
    - TLF_CONFIG.json [paths][raw_dictionary_full]
    - TLF_CONFIG.json [processing][words_per_chunk]

Outputs:
    - raw/dict/2-jsonl/dictionary_part{idx}.jsonl

Usage:
    python code/dict/02_chunk_data.py
"""

import os
import sys
import json
import fileinput
from utility import * 

# Universal Logger Setup
logger = setup_logger("Chunking", step_name="chunking")

INPUT_FILE = os.path.join(REPO_ROOT, TLF_CONFIG['paths']['raw_dictionary_full'])
OUTPUT_DIR = os.path.join(REPO_ROOT, TLF_CONFIG['paths']['jsonl_input_dir'])
WORDS_PER_CHUNK = TLF_CONFIG['processing']['words_per_chunk']

def chunk_data() -> None:
    """
    Sequentially reads the raw extraction and writes out chunks based on WORDS_PER_CHUNK.
    Ensures that trailing empty files are cleaned up.
    
    Args:
        None (Uses global CONFIG and paths).

    Returns:
        None (Writes files to OUTPUT_DIR).
    """
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Error: {INPUT_FILE} does not exist. Please run 01_extract_db.py first.")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    file_idx = 1
    line_count = 0
    out_file_path = os.path.join(OUTPUT_DIR, f"dictionary_part{file_idx:03d}.jsonl")
    out_f = open(out_file_path, 'w', encoding='utf-8')
    
    logger.info(f"Chunking {INPUT_FILE} into {WORDS_PER_CHUNK}-word chunks...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            if not line.strip(): continue
            out_f.write(line)
            line_count += 1
            
            if line_count >= WORDS_PER_CHUNK:
                out_f.close()
                logger.info(f"Created chunk {file_idx}: {out_file_path}")
                file_idx += 1
                line_count = 0
                out_file_path = os.path.join(OUTPUT_DIR, f"dictionary_part{file_idx:03d}.jsonl")
                out_f = open(out_file_path, 'w', encoding='utf-8')
                
    out_f.close()
    
    # Cleanup empty last file if it was freshly opened
    if line_count == 0 and os.path.exists(out_file_path):
        os.remove(out_file_path)
        
    logger.info("Chunking complete!")

if __name__ == "__main__":
    chunk_data()
