"""
Pipeline Utilities & Shared Infrastructure

Purpose:
    Provides standardized logging, configuration loading, and 
    JSONL I/O utilities used across all pipeline stages.

Functions:
    - setup_logger: Creates a dual-stream (console + file) logger.
    - load_config: Centralized tlf_config.json parser with caching.
    - createJSONL: Safely exports data to newline-delimited JSON.
"""

import json
import os
import logging
import sys
import re
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
LOGS_DIR = os.path.join(REPO_ROOT, "logs")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def normalize_dictionary_text(text: str) -> str:
    """
    Standardizes and cleans raw dictionary text to prevent numerical instability 
    during fine-tuning. Fixes spaced numbering, cleans references, and 
    unescapes HTML entities.
    """
    if not text: return ""
    
    # 1. Unescape HTML (e.g., &amp; -> &)
    import html
    text = html.unescape(text)
    
    # 2. Standardize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # 3. Fix spaced numbering (e.g. 4, 32, 000 -> 4,32,000)
    # Match digit followed by comma+space followed by digit
    text = re.sub(r'(\d),\s+(\d)', r'\1,\2', text)
    
    # 4. Fix multiple periods and other redundant punctuation
    text = re.sub(r'\.\.+', '.', text)
    text = re.sub(r',,+', ',', text)
    
    # 5. Clean reference legacy markers (convert + to space)
    # Only if it's within [bracketed] references or similar patterns
    # Pragmatic fix: just replace + globally if it's unlikely to be intentional math
    text = text.replace('+', ' ')
    
    return text.strip()

_cached_config: Optional[Dict[str, Any]] = None

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Universal configuration loader for the TLF-7B-LLM-01 pipeline.
    
    Args:
        config_path (Optional[str]): Absolute or relative path to tlf_config.json. 
            If None, searches in script directory or one level up.

    Returns:
        Dict[str, Any]: The parsed configuration dictionary.

    Notes:
        Caches the config in memory after the first load to improve performance.
    """
    global _cached_config
    if _cached_config and config_path is None:
        return _cached_config
        
    if config_path is None:
        # Predictable location: same directory as utility.py or one level up
        config_path = os.path.join(SCRIPT_DIR, "TLF_CONFIG.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(SCRIPT_DIR, "..", "TLF_CONFIG.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        _cached_config = json.load(f)
        
    load_dotenv(os.path.join(REPO_ROOT, ".env"))

    if os.getenv("DB_CONNECTION_STRING") and "database" in _cached_config:
        _cached_config["database"]["connection_string"] = os.getenv("DB_CONNECTION_STRING")
        
    if os.getenv("HF_TOKEN") and "finetuning" in _cached_config:
        _cached_config["finetuning"]["hf_token"] = os.getenv("HF_TOKEN")

    return _cached_config

def setup_logger(name: str, log_file: Optional[str] = None, level: Optional[int] = None, step_name: Optional[str] = None) -> logging.Logger:
    """
    Initializes a logger with both Stream (console) and File handlers.
    
    Args:
        name (str): Unique identifier for the logger instance (e.g., 'Extraction').
        log_file (Optional[str]): Path to the output log file. 
        level (Optional[int]): Standard logging level (e.g., logging.INFO). 
            If None, defaults to config['logging']['default_level'].
        step_name (Optional[str]): Key in tlf_config.json [logging][steps] to auto-resolve log_file path.
    
    Returns:
        logging.Logger: The configured logger instance providing dual-stream output.
    """
    config = load_config()
    log_dir = config.get('logging', {}).get('log_dir', 'logs')
    
    # Auto-resolve log_file if step_name is provided
    if step_name and log_file is None:
        step_filename = config.get('logging', {}).get('steps', {}).get(step_name)
        if step_filename:
            log_file = os.path.join(REPO_ROOT, log_dir, step_filename)

    if level is None:
        lvl_str = config.get('logging', {}).get('default_level', 'INFO')
        level = getattr(logging, lvl_str.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Stream Handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def createJSONL(word_list: List[Dict[str, Any]], output_file: str) -> None:
    """
    Exports a list of python dictionaries to a newline-delimited JSON (JSONL) file.
    
    Args:
        word_list (List[Dict[str, Any]]): A list of word objects to be serialized.
        output_file (str): The destination file path.

    Returns:
        None
    """
    logger = setup_logger("Utility-IO")
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in word_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"Successfully created JSONL: {output_file}")
    except Exception as e:
        logger.error(f"Failed to create JSONL: {e}")

def save_nan_diagnostic(iteration: int, record: Dict[str, Any], note: str = "NaN loss detected") -> str:
    """
    Saves problematic training record data to a debug file for investigation.
    """
    import datetime
    debug_path = os.path.join(REPO_ROOT, "nan_batch_debug.json")
    try:
        diag_data = {
            "iteration": iteration,
            "timestamp": datetime.datetime.now().isoformat(),
            "record": record,
            "note": note,
            "tip": "Reduce max_seq_length, lora_r, or check for numerical spikes in definitions."
        }
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(diag_data, f, indent=2, ensure_ascii=False)
        return debug_path
    except Exception as e:
        return f"Error saving diagnostic: {e}"

def log_skipped_record(iteration: Union[int, str], record: Dict[str, Any], output_dir: str) -> None:
    """
    Permanently blacklists a problematic iteration and logs its details to a 
    dedicated file for later troubleshooting.
    """
    import datetime
    blacklist_path = os.path.join(output_dir, "nan_blacklist.json")
    
    config = load_config()
    model_id = config.get("finetuning", {}).get("model_id", "unknown_model")
    safe_model_id = model_id.replace('/', '_')
    log_folder = os.path.join(LOGS_DIR, safe_model_id)
    os.makedirs(log_folder, exist_ok=True)
    log_path = os.path.join(log_folder, "skipped_records.log")
    
    # 1. Update JSON Blacklist for automatic filtering
    blacklist = []
    if os.path.exists(blacklist_path):
        try:
            with open(blacklist_path, 'r') as f:
                blacklist = json.load(f)
        except: blacklist = []
    
    if iteration not in blacklist:
        blacklist.append(iteration)
        with open(blacklist_path, 'w') as f:
            json.dump(blacklist, f)

    # 2. Detailed Log for Troubleshooting
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"[{timestamp}] ITERATION: {iteration}\n"
        f"RECORD SNIPPET: {str(record.get('text', 'N/A'))[:500]}...\n"
        f"FULL RECORD DATA: {json.dumps(record, ensure_ascii=False)}\n"
        f"{'-'*80}\n"
    )
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(log_entry)

def get_nan_blacklist(output_dir: str) -> List[Union[int, str]]:
    """Retrieves the list of iterations/batches to skip."""
    blacklist_path = os.path.join(output_dir, "nan_blacklist.json")
    if not os.path.exists(blacklist_path):
        return []
    try:
        with open(blacklist_path, 'r') as f:
            return json.load(f)
    except:
        return []

TLF_CONFIG = load_config()