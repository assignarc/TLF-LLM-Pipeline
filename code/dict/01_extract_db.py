from code.dict.utility import TLF_CONFIG
import os
import sys
import json
import pyodbc
import re
import html
import csv
from itertools import groupby
from bs4 import BeautifulSoup

from utility import * 

# Universal Logger Setup
logger = setup_logger("Extraction", step_name="extraction")

# Load References map
references_file = os.path.join(REPO_ROOT, TLF_CONFIG.get('paths', {}).get('references', 'dict/references.json'))
reference_titles = {}
if os.path.exists(references_file):
    with open(references_file, 'r', encoding='utf-8') as f:
        refs_data = json.load(f)
        for r in refs_data:
            if "Code" in r and "Title" in r:
                reference_titles[r["Code"]] = r["Title"]

CONN_STR = TLF_CONFIG['database']['connection_string']
OUTPUT_FILE = os.path.join(REPO_ROOT, TLF_CONFIG['paths']['jsonl_output_dir'], TLF_CONFIG['paths']['raw_dictionary_full'])
REPORT_FILE = os.path.join(REPO_ROOT, TLF_CONFIG['paths']['jsonl_output_dir'], TLF_CONFIG['paths']['reference_extraction_report'])

def fetch_data():
    try:
        conn = pyodbc.connect(CONN_STR)
        cursor = conn.cursor()
        logger.info("Connected to database successfully.")
        
        logger.info("Fetching full database... This may take a moment.")
        query = """
        SELECT
            w.WordId, w.WordText, w.Language as WordLang, w.WordTextAlternate, w.Syllables,
            m.MeaningId, m.MeaningText, m.Type, m.TypeGender, m.Language as MeaningLang, m.SeqNo,
            s.SourceShortText, s.SourceLongText, s.SourceLanguages
        FROM Words w
        LEFT JOIN Meanings m ON w.WordId = m.WordId
        LEFT JOIN Sources s ON m.SourceId = s.SourceId
        ORDER BY w.WordId, s.SourceShortText, m.SeqNo
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if not rows:
            logger.info("No records found.")
            return

        meaning_ids = list(set([str(r.MeaningId) for r in rows if r.MeaningId]))
        extensions_map = {}
        links_map = {}
        found_references = {}
        
        if meaning_ids:
            logger.info("Fetching all Extensions...")
            query_extensions = "SELECT MeaningId, ExtensionType, WordList, Details FROM Extensions"
            cursor.execute(query_extensions)
            for ext in cursor.fetchall():
                m_id = str(ext.MeaningId)
                if m_id not in extensions_map:
                    extensions_map[m_id] = []
                extensions_map[m_id].append({
                    "ExtensionType": ext.ExtensionType,
                    "WordList": ext.WordList,
                    "Details": ext.Details
                })
                
            logger.info("Fetching all Extension Links...")
            query_links = "SELECT MeaningId, WordList FROM ExtensionLinks"
            cursor.execute(query_links)
            for lnk in cursor.fetchall():
                m_id = str(lnk.MeaningId)
                if m_id not in links_map:
                    links_map[m_id] = []
                    
                if lnk.WordList:
                    clean_words = [w.strip() for w in lnk.WordList.replace('|', ',').split(',') if w.strip()]
                    if clean_words:
                        links_map[m_id].append({
                            "WordList": ", ".join(clean_words)
                        })

        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
        logger.info(f"Writing everything to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for word_id, group in groupby(rows, key=lambda x: str(x.WordId)):
                group_list = list(group)
                first = group_list[0]
                
                word_json = {
                    "metadata": {
                        "Language": first.WordLang if first.WordLang else "sa",
                        "WordText": first.WordText,
                        "WordTextAlternate": first.WordTextAlternate,
                        "Syllables": first.Syllables,
                        "TypeClass": "dictionary_entry"
                    },
                    "content": []
                }
                
                for row in group_list:
                    if not row.MeaningId:
                        continue 
                        
                    raw_meaning = row.MeaningText if row.MeaningText else ""
                    
                    references = []
                    ref_regex = TLF_CONFIG.get('processing', {}).get('dictionary_ref_regex', r'<a\s+[^>]*href=["\']/dictionary/([^/]+)/text\?ref=([^"\']+)["\'][^>]*>')
                    ref_matches = re.finditer(ref_regex, raw_meaning)
                    for match in ref_matches:
                        internal_code = match.group(1)
                        ref_obj = {
                            "InternalCode": internal_code,
                            "ReferenceCode": match.group(2).rstrip('.')
                        }
                        if internal_code in reference_titles:
                            title = reference_titles[internal_code]
                            ref_obj["Title"] = title
                            found_references[internal_code] = title
                        else:
                            found_references[internal_code] = "NOT FOUND"
                        references.append(ref_obj)
                        
                    clean_meaning = html.unescape(raw_meaning)
                    clean_meaning = BeautifulSoup(clean_meaning, "html.parser").get_text(separator=" ")
                    clean_meaning = re.sub(r'\s+', ' ', clean_meaning).strip()
                    
                    meaning = {
                        "MeaningText": clean_meaning,
                        "Type": row.Type,
                        "TypeGender": row.TypeGender,
                    }
                    if references:
                        meaning["References"] = references
                    if row.SeqNo is not None:
                        meaning["SeqNo"] = int(row.SeqNo)
                    if getattr(row, 'MeaningLang', None):
                        meaning["Language"] = row.MeaningLang
                    if row.SourceShortText or row.SourceLongText:
                        meaning["Source"] = {
                            "ShortText": row.SourceShortText,
                            "LongText": row.SourceLongText
                        }
                        if getattr(row, 'SourceLanguages', None):
                            meaning["Source"]["Languages"] = row.SourceLanguages
                    
                    m_id = str(row.MeaningId)
                    if m_id in extensions_map:
                        meaning["Extensions"] = extensions_map[m_id]
                    if m_id in links_map:
                        meaning["ExtensionLinks"] = links_map[m_id]
                        
                    word_json["content"].append(meaning)
                
                f.write(json.dumps(word_json, ensure_ascii=False) + '\n')
                
        if found_references:
            os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
            with open(REPORT_FILE, 'w', encoding='utf-8') as rf:
                rf.write("Code,Mapping\n")
                for code in sorted(found_references.keys()):
                    rf.write(f"{code},{found_references[code]}\n")
            logger.info(f"Logged comprehensive reference mapping to {REPORT_FILE}")
            
        logger.info(f"Extraction Step 1 Complete! Data saved to {OUTPUT_FILE}")
                
    except Exception as e:
        logger.info(f"Error during extraction: {e}")

if __name__ == "__main__":
    fetch_data()
