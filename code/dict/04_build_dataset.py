"""
Step 4: SFT Dataset Generation (JSONL to LLaMA-3.1 Prompt)

Purpose:
    Transforms processed dictionary entries into the LLaMA-3.1 instruction format:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{persona}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{target_response}<|eot_id|>
    Handles suffix substitution, usage extraction, and multi-task prompt mapping.
    Includes systemic normalization and deduplication.

Inputs:
    - raw/dict/3-chunks/dictionary_processed_*.jsonl
    - TLF_CONFIG.json [system_prompts] (Task-specific personas)
    - TLF_CONFIG.json [processing] (Regex for substitutions/usages)

Outputs:
    - raw/dict/4-prompts/instruction_dataset_{idx}.jsonl
    - raw/dict/4-prompts/substitutions_log.txt
"""

import os
import json
import glob
import re
import argparse
import hashlib
from typing import Set, Dict, Any
from utility import * 

def validate_content(text: str) -> bool:
    """Heuristic to catch potentially problematic training examples."""
    # 1. Minimum content check
    if not text or len(text.strip()) < 5:
        return False
        
    # 2. Check for empty definitions (e.g. "is defined as: .")
    clean = text.strip().lower()
    if "is defined as: ." in clean or "is defined as: ." in clean:
        return False
        
    # 3. Check for specific problematic keywords that leak from SQL/NaN
    for kw in ["<null>", "undefined", "n/a", "अभिसम्भृत"]:
        if kw in clean:
            return False
            
    # Whole-word check for 'nan' to avoid matching 'banana', 'finance', etc.
    import re
    if re.search(r'\bnan\b', clean):
        return False
        
    # 4. Check for runaway long strings that would cause OOM
    if len(text) > 10000:
        return False

    return True

def main() -> None:
    """
    Orchestrates the conversion of cleaned JSON dictionary nodes into
    Supervised Fine-Tuning (SFT) instruction pairs.
    """
    
    # Universal Logger Setup
    logger = setup_logger("Dataset-Building", step_name="dataset_building")
    
    mappings_file = os.path.join(REPO_ROOT, TLF_CONFIG['paths']['extension_mappings'])
    with open(mappings_file, 'r', encoding='utf-8') as f:
        ext_mappings = json.load(f).get("Extensions", {})
        
    input_dir = os.path.join(REPO_ROOT, TLF_CONFIG['paths']['jsonl_output_dir'])
    output_dir = os.path.join(REPO_ROOT, TLF_CONFIG['paths']['prompts_output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    log_name = TLF_CONFIG.get('paths', {}).get('substitutions_log', 'substitutions_log.txt')
    log_file_path = os.path.join(output_dir, log_name)
    
    max_chunk_size = TLF_CONFIG.get('processing', {}).get('words_per_chunk', 5000)
    max_citations = TLF_CONFIG.get('processing', {}).get('max_citations', 10)
    max_record_length = TLF_CONFIG.get('processing', {}).get('max_record_length', 15000)
    
    processed_files = glob.glob(os.path.join(input_dir, "dictionary_processed_*.jsonl"))
    if not processed_files:
        logger.error(f"No processed files found in {input_dir}")
        return

    instruction_count = 0
    file_idx = 1
    
    out_f = None
    log_f = open(log_file_path, 'w', encoding='utf-8')
    log_f.write("--- SUFFIX SUBSTITUTIONS LOG ---\n")
    
    # Track "seen" instructions to prevent redundant near-identical training pairs
    # per word concepts. Hash set is used for memory efficiency.
    seen_hashes: Set[str] = set()
    
    def open_next_file() -> None:
        """Handlers rotation of output instruction files."""
        nonlocal out_f, file_idx, instruction_count
        if out_f: out_f.close()
        out_path = os.path.join(output_dir, f"instruction_dataset_{file_idx:03d}.jsonl")
        out_f = open(out_path, 'w', encoding='utf-8')
        logger.info(f"Writing to {out_path}...")
        file_idx += 1
        instruction_count = 0
        
    def write_instruction(sys_prompt: str, usr_prompt: str, target: str) -> None:
        """Standardizes and writes a single instruction block with deduplication."""
        nonlocal instruction_count
        
        # 0. Basic Validation
        if not validate_content(target):
            return

        # 1. Normalization of target text (redundant but safe)
        target = normalize_dictionary_text(target)
        
        # 2. Semantic Deduplication
        # Use a combination of user prompt and a significant portion of target to detect duplicates
        # We Hash the combination to save memory.
        concept_key = f"{usr_prompt}|{target[:500]}".strip()
        h = hashlib.md5(concept_key.encode('utf-8')).hexdigest()
        if h in seen_hashes:
            return
        seen_hashes.add(h)
        
        if out_f is None or instruction_count >= max_chunk_size:
            open_next_file()
            
        # Zero-Persona Mode: Dynamically select format based on configured exact model_id
        model_id = TLF_CONFIG.get("finetuning", {}).get("model_id", "")
        templates = TLF_CONFIG.get("prompt_templates", {})
        
        # Determine active template (default to llama-3.1 if no match)
        active_template = templates.get(model_id, templates.get("meta-llama/Meta-Llama-3.1-8B", {"format": "<s>[INST] {user_prompt} [/INST] {target} </s>", "truncation_chars": 7}))
        
        fmt = active_template.get("format", "<s>[INST] {user_prompt} [/INST] {target} </s>")
        t_chars = active_template.get("truncation_chars", 7)
        
        text = fmt.format(user_prompt=usr_prompt, target=target)
        
        # 3. Safeguard: Truncate if exceeds limits or configured safety
        if len(text) > max_record_length:
            logger.warning(f"Record for '{usr_prompt[:30]}...' is long ({len(text)}). Truncating to {max_record_length}.")
            tail = fmt.split("{target}")[-1] if "{target}" in fmt else " </s>"
            text = text[:max_record_length - len(tail) - 5] + " ... " + tail

        out_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        instruction_count += 1

    for file_path in sorted(processed_files):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                word_obj = json.loads(line)
                meta = word_obj.get("metadata", {})
                content = word_obj.get("content", [])
                
                word_text = meta.get("WordText", "")
                word_lang = meta.get("Language", "")
                alt_text = meta.get("WordTextAlternate")
                
                for meaning in content:
                    raw_text = meaning.get("MeaningText", "")
                    
                    # 1. Suffix Substitution
                    clean_text = raw_text
                    if TLF_CONFIG.get('processing', {}).get('enable_suffix_substitution', False):
                        def replace_suffix(m):
                            subbed = word_text + m.group(1)
                            log_f.write(f"Word: {word_text} | Pattern: ०{m.group(1)} -> {subbed}\n")
                            return subbed
                            
                        suffix_regex = TLF_CONFIG.get('processing', {}).get('suffix_substitution_regex', r'०([^\W\d_]+)')
                        clean_text = re.sub(suffix_regex, replace_suffix, clean_text)
                    
                    # 2. Extract Usages
                    usages = []
                    ex_regex = TLF_CONFIG.get('processing', {}).get('usage_ex_regex', r'\bEx\.\s+')
                    ex_split = re.split(ex_regex, clean_text, maxsplit=1)
                    if len(ex_split) > 1:
                        clean_text = ex_split[0].strip()
                        usages.append(ex_split[1].strip())
                        
                    quote_regex = TLF_CONFIG.get('processing', {}).get('usage_quote_regex', r"'([^']+)'\s*-\s*([^.]+)\.")
                    quote_dash_matches = re.finditer(quote_regex, clean_text)
                    for qm in quote_dash_matches:
                        usages.append(f"\"{qm.group(1).strip()}\" - Source: {qm.group(2).strip()}")
                    clean_text = re.sub(quote_regex, "", clean_text).strip()
                    
                    # Build References
                    refs = meaning.get("References", [])
                    ref_text = ""
                    if refs:
                        ref_parts = []
                        for r in refs[:max_citations]:
                            title = r.get('Title', r.get('InternalCode', ''))
                            ref_c = r.get('ReferenceCode', '')
                            ref_parts.append(f"{title} [{ref_c}]")
                        ref_text = "Citations: " + ", ".join(ref_parts) + ("..." if len(refs) > max_citations else "") + "."
                        
                    # TASK A: Definition & Morphology
                    sys_prompts = TLF_CONFIG.get('system_prompts', {})
                    sys_a = sys_prompts.get('task_a_morphology', "You are an expert bilingual lexicographer for Indic languages.")
                    
                    part_info = f" ({meaning['part_index']}/{meaning['total_parts']})" if "part_index" in meaning else ""
                        
                    usr_a = f"Def:{part_info} {word_text}"
                    alt_info = f" (also: {alt_text})" if alt_text else ""
                    target_a_parts = [f"\"{word_text}\"{alt_info} Def: {clean_text}."]
                    
                    if usages: target_a_parts.append("Ex: " + " | ".join(usages))
                    
                    managed = meaning.get("ManagedType", [])
                    if managed: target_a_parts.append(f"Tags: {', '.join(managed)}.")
                    if ref_text: target_a_parts.append(ref_text)
                    
                    write_instruction(sys_a, usr_a, " ".join(target_a_parts))
                    
                    # Other tasks follow same logic (Deduplication handles repeats)
                    ext_details = meaning.get("Extensions", [])
                    translations = []
                    synonyms = []
                    
                    for ed in ext_details:
                        etype = ed.get("ExtensionType", "")
                        wl = ed.get("WordList", "")
                        if not wl: continue
                        if etype.startswith("iwn.lang."):
                            translations.append(f"{etype.split('.')[-1].upper()}: {wl}")
                        elif "SYNONYM" in etype:
                            synonyms.append(wl)
                            
                    if translations:
                        usr_b = f"Trans:{part_info} {word_text}"
                        target_b = f"Translations: {', '.join(translations)}."
                        if ref_text: target_b += " " + ref_text
                        write_instruction(None, usr_b, target_b)
                        
                    if synonyms:
                        usr_c2 = f"Syn:{part_info} {word_text}"
                        target_c2 = f"Synonyms: {', '.join(synonyms)}."
                        if ref_text: target_c2 += " " + ref_text
                        write_instruction(None, usr_c2, target_c2)
                        
                    # TASK E: References
                    if refs:
                        usr_e = f"Cite:{part_info} {word_text}"
                        target_e_parts = [f"\"{word_text}\" Def: {clean_text}."]
                        for r in refs[:max_citations]:
                            title = r.get('Title', r.get('InternalCode', ''))
                            ref_c = r.get('ReferenceCode', '')
                            target_e_parts.append(f"- Source: {title} [{ref_c}]")
                        write_instruction(None, usr_e, "\n".join(target_e_parts))

    if out_f: out_f.close()
    log_f.close()
    logger.info(f"Dataset building complete! Saved to {output_dir}")

if __name__ == "__main__":
    main()
