"""
Step 7: Model Publisher & Card Generator

Purpose:
    Prepares the fine-tuned model for HuggingFace Hub. Generates a 
    professional Model Card (README.md) and handles the upload of 
    adapter weights and configurations.

Inputs:
    - TLF_CONFIG.json (For hyperparameters and metadata)
    - raw/models/ (Source of adapter weights)
    - logs/ (Source of training telemetry)

Outputs:
    - README.md (Model Card formatted for HF Hub)
    - HF Repository upload (Optional)

Usage:
    python code/dict/07_publish_model.py [--dry-run] [--repo_id username/TLF-7B-LLM-01]
"""

import os
import json
import glob
import argparse
from datetime import datetime
from utility import *

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

def generate_model_card(config: dict, stats: dict = None) -> str:
    """
    Constructs a premium Markdown model card based on the current pipeline state.
    """
    ft_config = config.get("finetuning", {})
    inf_config = config.get("inference", {})
    model_id = ft_config.get("model_id", "Unknown")
    model_name = config.get("model_name", "TLF-7B-LLM-01")
    engine = ft_config.get("engine", "hf")
    
    # Extract prompt template for the model
    prompt_templates = config.get("prompt_templates", {})
    template_info = prompt_templates.get(model_id, {})
    template = template_info.get("format", "<s>[INST] {user_prompt} [/INST] {target} </s>")
    
    # Handle the specific Sarvam logic we found
    if "sarvam-1" in model_id.lower():
        template_display = "<s>[INST] <<SYS>>\\n{system_prompt}\\n<</SYS>>\\n\\n{query} [/INST]"
    else:
        template_display = template.replace("{target}", "...")

    card = f"""---
license: apache-2.0
base_model: {model_id}
library_name: peft
tags:
- indic-nlp
- dictionary
- sanskrit
- marathi
- hindi
- sft
- lora
---

# {model_name}

## Model Description
This model is a fine-tuned version of [{model_id}](https://huggingface.co/{model_id}) specialized for **Bilingual Indic Lexicography**. 
It has been trained to provide structured morphological breakdowns, definitions, and regional translations for Sanskrit and other Indian regional languages.

The training data was ingested through the **TLF Mega-Pipeline**, integrating structured dictionary databases (MSSQL) with unstructured regional texts to improve grammar and stylistic intelligence.

## Intended Use
- **Dictionary Lookups**: Providing high-accuracy definitions and etymologies.
- **Morphological Analysis**: Breaking down complex Sanskrit/Indic root words.
- **Regional Translation**: Translating word concepts across Marathi, Hindi, and English.

## Training Hyperparameters
The following hyperparameters were used during training:
- **Engine**: {engine.upper()}
- **Learning Rate**: {ft_config.get("learning_rate", "2e-5")}
- **Batch Size**: {ft_config.get("batch_size", 1)}
- **Gradient Accumulation**: {ft_config.get("gradient_accumulation_steps", 64)}
- **Optimizer**: {ft_config.get("optim", "adamw_torch")}
- **LR Scheduler**: {ft_config.get("lr_scheduler_type", "cosine")}
- **LoRA R**: {ft_config.get("lora_r", 32)}
- **LoRA Alpha**: {ft_config.get("lora_alpha", 16)}
- **Max Sequence Length**: {ft_config.get("max_seq_length", 1024)}

## Prompt Template
To achieve the intended structured output, use the following prompt format:

```text
{template_display}
```

## Inference Example
### Using MLX (Apple Silicon)
```python
import mlx_lm
model, tokenizer = mlx_lm.load("YourAccount/{model_name}")

prompt = "Provide a comprehensive morphological breakdown for: 'Abacus'"
# Use Sarvam/Llama template logic here
response = mlx_lm.generate(model, tokenizer, prompt=prompt)
print(response)
```

### Using Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("{model_id}")
model = PeftModel.from_pretrained(base_model, "YourAccount/{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_id}")

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

## Citation & Credits
- **TLF Framework**: Architected for Unified Indic LLM Fine-tuning.
- **Data Source**: Custom Dictionary & Regional Text Corpus.
"""
    return card

def main():
    parser = argparse.ArgumentParser(description="Publish TLF Model to HF Hub")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate the Model Card (README.md) without uploading.")
    parser.add_argument("--output", type=str, default="MODEL_CARD_PREVIEW.md",
                        help="Path to save the generated model card preview locally.")
    parser.add_argument("--repo_id", type=str, default=None,
                        help="HuggingFace repository ID to publish to (e.g. username/TLF-7B-LLM-01).")
    args = parser.parse_args()

    # 1. Setup Logger
    logger = setup_logger("Publisher", step_name="publishing")

    # 2. Discover the adapter directory under raw/models/
    #    Step 5 names it as <output_name>_<safe_model_id>; discover by looking for
    #    the folder containing the final adapters.safetensors rather than guessing the name.
    ft_config = TLF_CONFIG.get("finetuning", {})
    models_root = os.path.join(REPO_ROOT, "raw", "models")
    candidate_dirs = sorted(glob.glob(os.path.join(models_root, "*_sarvamai_sarvam-1"))) + \
                     sorted(glob.glob(os.path.join(models_root, "*_sarvam*"))) + \
                     sorted(glob.glob(os.path.join(models_root, "*")))
    model_dir = None
    for d in candidate_dirs:
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "adapters.safetensors")):
            model_dir = d
            break
    if model_dir is None:
        model_dir = os.path.join(models_root, "TLF-7B-MLX-01_sarvamai_sarvam-1")  # fallback
    logger.info(f"Resolved model directory: {model_dir}")

    # 3. Model Card: use existing README.md in model dir, or auto-generate one
    readme_path = os.path.join(model_dir, "README.md")

    if os.path.exists(readme_path):
        # User has manually edited the card — leave it completely untouched.
        with open(readme_path, encoding="utf-8") as f:
            card_content = f.read()
        logger.info(f"[MODEL CARD] Using existing README.md: {readme_path}")
        print(f"\n[MODEL CARD] Found manually edited README.md — using it as-is.")
        print(f"  {readme_path}")
    else:
        # Nothing there yet — auto-generate and write into the model directory.
        logger.info("[MODEL CARD] No README.md in model directory — auto-generating...")
        card_content = generate_model_card(TLF_CONFIG)

        os.makedirs(model_dir, exist_ok=True)
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(card_content)
        logger.info(f"[MODEL CARD] Auto-generated README.md written to: {readme_path}")

        # Also write a preview at repo root so it is easy to open and edit.
        preview_path = os.path.join(REPO_ROOT, args.output)
        with open(preview_path, "w", encoding="utf-8") as f:
            f.write(card_content)
        logger.info(f"[MODEL CARD] Preview copy: {preview_path}")
        print(f"\n[MODEL CARD] Auto-generated.")
        print(f"  Canonical file (edit this): {readme_path}")
        print(f"  Preview copy (read-only):   {preview_path}")

    # 4. Dry-run: show what would happen, then stop.
    if args.dry_run:
        logger.info("[DRY-RUN] Upload skipped.")
        print("\n[DRY-RUN] Nothing uploaded.")
        print("  When ready, run without --dry-run and add --repo_id <hf-username>/<repo-name>")
        return

    # 5. Validate repo_id before attempting upload.
    if not args.repo_id:
        logger.warning("No --repo_id provided. Nothing uploaded.")
        print("[INFO] Pass --repo_id username/repo-name to upload to HuggingFace Hub.")
        return

    if not HF_HUB_AVAILABLE:
        logger.error("huggingface_hub not installed. Run: pip install huggingface-hub")
        print("[ERROR] pip install huggingface-hub  — then retry.")
        return

    # 6. Ensure the HuggingFace repo exists.
    logger.info(f"Uploading to HuggingFace Hub: {args.repo_id}")
    print(f"\n[UPLOAD] Target: https://huggingface.co/{args.repo_id}")

    try:
        create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
        logger.info(f"Repository ensured: {args.repo_id}")
    except Exception as e:
        logger.error(f"Failed to create/verify repository: {e}")
        raise

    if not os.path.isdir(model_dir):
        logger.error(f"Model directory not found: {model_dir}")
        print(f"[ERROR] Expected adapter directory at: {model_dir}")
        print("  Run Step 5 first to generate adapter weights.")
        return

    # 7. Upload entire model directory (adapters + mlx_config.yaml + README.md).
    logger.info(f"Uploading: {model_dir}")
    upload_folder(
        repo_id=args.repo_id,
        folder_path=model_dir,
        repo_type="model",
        commit_message=f"TLF-7B-LLM-01: Upload LoRA adapters ({datetime.now().strftime('%Y-%m-%d')})",
        ignore_patterns=["*.log", "__pycache__", ".DS_Store", "data/"],
    )

    logger.info(f"Upload complete: https://huggingface.co/{args.repo_id}")
    print(f"\n[PUBLISHED] https://huggingface.co/{args.repo_id}")
    print("-" * 60)
    print("Next steps:")
    print("  - Update article links to point to the HuggingFace URL")
    print("  - Run Step 9 to export a GGUF for llama.cpp deployment")

if __name__ == "__main__":
    main()
