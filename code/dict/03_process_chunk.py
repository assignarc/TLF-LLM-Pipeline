"""
Step 3: Chunk Processing & Meaning Merging

Purpose:
    Normalizes dictionary entries by merging multiple sequential 'Meaning' nodes
    that belong to the same source. Extracts morphological 'ManagedType' tags.

Inputs:
    - raw/dict/2-jsonl/dictionary_part{idx}.jsonl
    - TLF_CONFIG.json [paths][mapping_file] (Type mapping)
    - TLF_CONFIG.json [processing] (Regex cleaning params)

Outputs:
    - raw/dict/3-chunks/dictionary_processed_{part_num}.jsonl

Usage:
    python code/dict/03_process_chunk.py --input_file <path> --output_file <path>
"""

import json
import os
import re
import argparse
import sys
from typing import Set, Optional, Dict, Any, List
from utility import * 

# Universal Logger Setup
logger = setup_logger("Processing", step_name="processing")

MAPPING_FILE = os.path.join(REPO_ROOT, TLF_CONFIG['paths']['mapping_file'])

with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
    mapping = json.load(f)

type_map = mapping.get('Types', {})
gender_map = mapping.get('TypeGenders', {})

exclude_keywords = TLF_CONFIG['processing']['exclude_keywords']
split_regex = TLF_CONFIG['processing']['split_regex']
seqno_extract_regex = TLF_CONFIG['processing']['seqno_extract_regex']

def extract_managed_types(t_raw: Optional[str], g_raw: Optional[str]) -> Set[str]:
    """
    Maps raw database types/genders to human-readable 'ManagedType' tokens
    using the provided mapping.json across split regex and exclusion filters.
    """
    t_key = (t_raw or "").strip()
    g_key = (g_raw or "").strip()
    t_mapped = type_map.get(t_key, "")
    g_mapped = gender_map.get(g_key, "")
    
    valid_tokens = set()
    
    def process_string(val):
        if not val: return
        tokens = re.split(split_regex, val)
        for tok in tokens:
            tok = tok.strip()
            if tok and not any(k in tok.lower() for k in exclude_keywords):
                valid_tokens.add(tok)
                
    process_string(t_mapped)
    process_string(g_mapped)
    
    return valid_tokens

def sort_key_seqno(m: Dict[str, Any]) -> int:
    """
    Helper to extract and return an integer Sequence Number for sorting meaning nodes.
    """
    seq = str(m.get('SeqNo', ''))
    matches = re.findall(seqno_extract_regex, seq)
    if matches: return int(matches[0])
    return 999999

def process_chunk(input_file: str, output_file: str) -> None:
    """
    Primary logic for consolidating split meaning entries. Merges text, citations,
    and extensions while preserving wordnet.indo specialized data.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    logger.info(f"Processing chunk: {input_file} -> {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as in_f, open(output_file, 'w', encoding='utf-8') as out_f:
        for line in in_f:
            if not line.strip(): continue
            try: word_entry = json.loads(line)
            except: continue
                
            new_content = []
            unmerged = []
            groups = {}
            
            chars_limit = TLF_CONFIG['processing'].get('max_chars_per_record_split', 10000)

            def get_split_meanings(m_item: Dict[str, Any]) -> List[Dict[str, Any]]:
                """Splits a single meaning item if its text exceeds chars_limit."""
                m_text = (m_item.get('MeaningText') or "").strip()
                if len(m_text) <= chars_limit:
                    return [m_item]
                
                logger.info(f"Splitting exceptionally long meaning ({len(m_text)} chars)")
                parts = []
                temp_parts = re.split(r'([।\n\.])', m_text)
                current_part = ""
                for i in range(0, len(temp_parts), 2):
                    segment = temp_parts[i]
                    sep = temp_parts[i+1] if i+1 < len(temp_parts) else ""
                    full_seg = segment + sep
                    if len(current_part) + len(full_seg) <= chars_limit:
                        current_part += full_seg
                    else:
                        if current_part: parts.append(current_part.strip())
                        if len(full_seg) > chars_limit:
                            for k in range(0, len(full_seg), chars_limit):
                                parts.append(full_seg[k:k+chars_limit].strip())
                            current_part = ""
                        else:
                            current_part = full_seg
                if current_part: parts.append(current_part.strip())
                
                new_items = []
                for idx, p in enumerate(parts):
                    if not p: continue
                    new_item = dict(m_item)
                    # Metadata Isolation: Only attach heavy extensions to the first part
                    if idx > 0:
                        new_item.pop('Extensions', None)
                        new_item.pop('ExtensionLinks', None)
                        new_item.pop('References', None)
                        
                    new_item['MeaningText'] = p
                    if len(parts) > 1:
                        new_item['part_index'] = idx + 1
                        new_item['total_parts'] = len(parts)
                    new_items.append(new_item)
                return new_items

            for m in word_entry.get('content', []):
                source_short = m.get('Source', {}).get('ShortText', '')
                has_extensions = bool(m.get('Extensions', []))
                
                # Apply splitting to EVERY item before unmerged/merged classification
                for split_m in get_split_meanings(m):
                    if 'wordnet.indo' in source_short or has_extensions:
                        unmerged.append(split_m)
                    else:
                        if source_short not in groups:
                            groups[source_short] = []
                        groups[source_short].append(split_m)
            
            for m in unmerged:
                m['ManagedType'] = sorted(list(extract_managed_types(m.get('Type'), m.get('TypeGender'))))
                new_content.append(m)
                
            for source, group in groups.items():
                if len(group) == 1:
                    m = group[0]
                    m['ManagedType'] = sorted(list(extract_managed_types(m.get('Type'), m.get('TypeGender'))))
                    new_content.append(m)
                else:
                    # Use the already split/expanded group nodes
                    group.sort(key=sort_key_seqno)
                    
                    meanings_limit = TLF_CONFIG['processing'].get('meanings_per_record', 4)
                    
                    subgroups = []
                    current_sub = []
                    current_chars = 0
                    
                    for m_item in group:
                        m_text = (m_item.get('MeaningText') or "").strip()
                        m_len = len(m_text)
                        
                        if current_sub and (len(current_sub) >= meanings_limit or (current_chars + m_len > chars_limit)):
                            subgroups.append(current_sub)
                            current_sub = []
                            current_chars = 0
                            
                        current_sub.append(m_item)
                        current_chars += m_len
                        
                    if current_sub:
                        subgroups.append(current_sub)
                        
                    for idx, subgroup in enumerate(subgroups):
                        merged_text_parts = []
                        seq_nos = []
                        managed_set = set()
                        all_extension_links = []
                        all_references = []
                        
                        for m_item in subgroup:
                            t = (m_item.get('Type') or "").strip()
                            g = (m_item.get('TypeGender') or "").strip()
                            
                            # Apply normalization to the meaning text
                            text = normalize_dictionary_text(m_item.get('MeaningText') or "")
                            
                            part = f"{t} {g} {text}".strip()
                            # Extra space cleanup after concatenation
                            part = re.sub(r'\s+', ' ', part)
                            merged_text_parts.append(part)
                            
                            seq = m_item.get('SeqNo')
                            if seq is not None and str(seq).strip() != "":
                                seq_nos.append(str(seq))
                                
                            managed_set.update(extract_managed_types(t, g))
                            
                            ext_links = m_item.get('ExtensionLinks', [])
                            if isinstance(ext_links, list):
                                for ext in ext_links:
                                    if ext not in all_extension_links:
                                        all_extension_links.append(ext)
                                        
                            refs = m_item.get('References', [])
                            if isinstance(refs, list):
                                for r in refs:
                                    if r not in all_references:
                                        all_references.append(r)
                                        
                        merged_m = dict(subgroup[0])
                        merged_m['MeaningText'] = "\n".join(merged_text_parts)
                        merged_m['SeqNo'] = ",".join(seq_nos)
                        merged_m['Type'] = ""
                        merged_m['TypeGender'] = ""
                        merged_m['ManagedType'] = sorted(list(managed_set))
                        
                        if len(subgroups) > 1:
                            merged_m['part_index'] = idx + 1
                            merged_m['total_parts'] = len(subgroups)
                            
                        if all_extension_links and idx == 0:
                            merged_m['ExtensionLinks'] = all_extension_links
                        if all_references and idx == 0:
                            merged_m['References'] = all_references
                        
                        new_content.append(merged_m)
                    
            new_content.sort(key=sort_key_seqno)
            word_entry['content'] = new_content
            out_f.write(json.dumps(word_entry, ensure_ascii=False) + "\n")
            
def main():
    parser = argparse.ArgumentParser(description="Process a single dictionary chunk.")
    parser.add_argument("--input_file", required=True, help="Path to input JSONL chunk")
    parser.add_argument("--output_file", required=True, help="Path to write processed JSONL chunk")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        logger.error(f"Error: Input file {args.input_file} does not exist.")
        sys.exit(1)
        
    process_chunk(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
