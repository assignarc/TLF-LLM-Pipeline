"""
Step 6: LoRA Adapter Interactive CLI Inference

Purpose:
    Provides a real-time terminal interface to test the fine-tuned LoRA 
    adapters. Supports both MLX and HuggingFace engines depending on 
    the current tlf_config.json setting.

Inputs:
    - tlf_config.json [finetuning][engine] (Target framework)
    - tlf_config.json [inference] (Sampling parameters)
    - --prompt (Optional single-string test)

Outputs:
    - Interactive '>>>' console for querying the model.

Usage:
    python code/dict/06_test_model.py [--prompt "Your word here"]
"""

import os
import argparse
import json
import sys
from utility import * 

def main() -> None:
    """
    Loads the base model + adapter, injects the system prompt, 
    and starts the interactive testing loop.
    
    Args:
        None (Uses argparse for input).

    Returns:
        None (Starts interactive loop).
    """
    parser = argparse.ArgumentParser(description=f"Test Fine-tuned LLaMA-2")
    parser.add_argument("--prompt", type=str, help="Single prompt to test execution.")
    parser.add_argument("--base-only", action="store_true", help="Test the base model without loading the fine-tuned adapter.")
    parser.add_argument("--compare", action="store_true", help="Run both the base model and the fine-tuned model simultaneously and log both.")
    args = parser.parse_args()

    ft_config = TLF_CONFIG.get("finetuning", {})
    inf_config = TLF_CONFIG.get("inference", {})

    engine = ft_config.get("engine", "hf")
    model_id = ft_config.get("model_id", "meta-llama/Llama-2-7b-hf")
    hf_token = ft_config.get("hf_token", None)
    if hf_token == "your_huggingface_token_here": hf_token = None
    
    # Determine the correct adapter directory based on engine
    base_output_dir = ft_config.get("output_dir", "raw/models/")
    safe_model_id = model_id.replace('/', '_')
    if engine == "mlx":
        TARGET_MODEL_NAME = TLF_CONFIG.get("model_name", "TLF-7B-MLX-01")
    else:
        TARGET_MODEL_NAME = TLF_CONFIG.get("model_name", "TLF-7B-HF-01")
        
    target_name = f"{TARGET_MODEL_NAME}_{safe_model_id}"
    output_dir = os.path.join(REPO_ROOT, base_output_dir, target_name)
    
    temperature = inf_config.get("temperature", 0.3)
    top_p = inf_config.get("top_p", 0.9)
    max_new_tokens = inf_config.get("max_new_tokens", 256)
    repetition_penalty = inf_config.get("repetition_penalty", 1.15)

    # Universal Logger Setup
    logger = setup_logger("Inference", step_name="inference")

    # Dedicated Test Log File
    from datetime import datetime
    test_log_file = os.path.join(REPO_ROOT, "logs", f"test_results_{TARGET_MODEL_NAME}.jsonl")
    
    def log_test_result(prompt, response, response_base=None):
        try:
            with open(test_log_file, "a", encoding="utf-8") as f:
                log_data = {"timestamp": datetime.now().isoformat(), "prompt": prompt, "response": response}
                if response_base is not None:
                    log_data["response_base"] = response_base
                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to log test result: {e}")

    logger.info(f"Engine: {engine.upper()}")
    logger.info(f"Using Adapter: {output_dir}")

    # --- ENGINE-SPECIFIC LOAD ---
    if engine == "mlx":
        try:
            import mlx_lm
            logger.info(f"Loading MLX Model: {model_id}...")
            if args.base_only:
                logger.info("Skipping adapter (--base-only). Loading base model natively.")
                model, tokenizer = mlx_lm.load(model_id)
            else:
                try:
                    logger.info(f"Merging base model with adapter: {output_dir}...")
                    model, tokenizer = mlx_lm.load(model_id, adapter_path=output_dir)
                    if args.compare:
                        logger.info("Loading secondary un-adapted base model for comparison...")
                        global base_mlx_model, base_mlx_tokenizer
                        base_mlx_model, base_mlx_tokenizer = mlx_lm.load(model_id)
                except Exception as e:
                    logger.warning(f"No valid adapter found at {output_dir} or merge failed. Falling back to base model. Error: {e}")
                    model, tokenizer = mlx_lm.load(model_id)
            
            def generate_func(text):
                prompt_templates = TLF_CONFIG.get("prompt_templates", {})
                template_info = prompt_templates.get(model_id, {})
                template = template_info.get("format", "<s>[INST] {user_prompt} [/INST] {target} </s>")
                
                # Check if it is sarvam-1 to handle special BOS/EOS tokens
                is_sarvam = "sarvam-1" in model_id.lower()
                
                sys_prompt = "You are an expert bilingual lexicographer for Indic languages."
                combined_text = sys_prompt + text
                
                # Construct precisely
                if is_sarvam:
                    # Surprisingly, Sarvam-1 tokenizer defines a Llama-2 [INST] template
                    formatted_prompt = f"<s>[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{text} [/INST]"
                else:
                    formatted_prompt = template.replace("{user_prompt}", combined_text).split("{target}")[0]
                
                import mlx_lm.sample_utils as su
                sampler = su.make_sampler(temp=temperature, top_p=top_p)
                logits_procs = su.make_logits_processors(repetition_penalty=repetition_penalty) if repetition_penalty != 1.0 else None
                resp_ft = mlx_lm.generate(
                    model, tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=max_new_tokens,
                    sampler=sampler,
                    logits_processors=logits_procs,
                    verbose=False
                )
                if resp_ft.startswith(formatted_prompt):
                    resp_ft = resp_ft[len(formatted_prompt):].strip()

                if args.compare and "base_mlx_model" in globals():
                    resp_base = mlx_lm.generate(
                        base_mlx_model, base_mlx_tokenizer,
                        prompt=formatted_prompt, max_tokens=max_new_tokens, sampler=sampler, logits_processors=logits_procs, verbose=False)
                    if resp_base.startswith(formatted_prompt):
                        resp_base = resp_base[len(formatted_prompt):].strip()
                    return resp_ft, resp_base
                return resp_ft
        except Exception as e:
            logger.info(f"\n[FATAL] MLX Initialization failed: {e}")
            return
    else:
        # HF ENGINE (Transformers/PEFT)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel
            
            device = ft_config.get("device", "auto")
            logger.info(f"Loading Base LLaMA-2 model: {model_id}...")
            if device == "mps":
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=torch.float16, device_map="mps", token=hf_token
                )
            else:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_id, quantization_config=bnb_config, device_map=device, token=hf_token
                )
            
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
            
            if not args.base_only and os.path.exists(os.path.join(output_dir, "adapter_config.json")):
                logger.info(f"Loading LoRA dictionary adapter from {output_dir}...")
                model = PeftModel.from_pretrained(base_model, output_dir)
            else:
                logger.info(f"\n[WARNING] No LoRA adapter found at {output_dir}. Using base model.")
                model = base_model
            model.eval()

            def generate_func(text):
                prompt_templates = TLF_CONFIG.get("prompt_templates", {})
                template_info = prompt_templates.get(model_id, {})
                template = template_info.get("format", "<s>[INST] {user_prompt} [/INST] {target} </s>")
                
                # Check if it is sarvam-1 to handle special BOS/EOS tokens
                is_sarvam = "sarvam-1" in model_id.lower()
                
                sys_prompt = "You are an expert bilingual lexicographer for Indic languages."
                combined_text = sys_prompt + text
                
                # Construct precisely
                if is_sarvam:
                    # Surprisingly, Sarvam-1 tokenizer defines a Llama-2 [INST] template
                    formatted_prompt = f"<s>[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{text} [/INST]"
                else:
                    formatted_prompt = template.replace("{user_prompt}", combined_text).split("{target}")[0]
                
                target_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
                inputs = tokenizer(formatted_prompt, return_tensors="pt").to(target_device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=max_new_tokens,
                        temperature=temperature, do_sample=True, top_p=top_p,
                        repetition_penalty=repetition_penalty
                    )
                resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                resp_base_txt = None
                if args.compare and hasattr(model, "disable_adapter"):
                    with torch.no_grad():
                        with model.disable_adapter():
                            out_base = model.generate(
                                **inputs, max_new_tokens=max_new_tokens,
                                temperature=temperature, do_sample=True, top_p=top_p,
                                repetition_penalty=repetition_penalty
                            )
                    resp_b = tokenizer.decode(out_base[0], skip_special_tokens=True)
                    cls_b = template.split("{target}")[-1].strip()
                    if cls_b and cls_b in resp_b: resp_b = resp_b.replace(cls_b, "")
                    strp_b = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
                    resp_base_txt = resp_b[len(strp_b):].strip() if resp_b.startswith(strp_b) else resp_b.strip()
                closing_syntax = template.split("{target}")[-1].strip()
                if closing_syntax and closing_syntax in resp:
                    resp = resp.replace(closing_syntax, "")
                stripped_prompt = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
                if resp.startswith(stripped_prompt):
                    resp = resp[len(stripped_prompt):].strip()
                else:
                    resp = resp.strip()
                
                if args.compare and resp_base_txt is not None:
                    return resp, resp_base_txt
                return resp
        except Exception as e:
            logger.info(f"\n[FATAL] HF Initialization failed: {e}")
            return

    if args.prompt:
        logger.info(f"\nPrompt: {args.prompt}")
        res = generate_func(args.prompt)
        if hasattr(args, "compare") and args.compare and isinstance(res, tuple):
            resp_ft, resp_base = res
            logger.info(f"[BASE MODEL]: {resp_base}")
            logger.info(f"[FINETUNED]: {resp_ft}\n")
            log_test_result(args.prompt, resp_ft, resp_base)
        else:
            logger.info(f"Response: {res}\n")
            log_test_result(args.prompt, res)
        return

    logger.info("=======================================================")
    logger.info(f"TLF-7B-LLM-01 CLI ({engine.upper()}) Engine Ready!")
    logger.info("Type 'quit' or 'exit' to terminate.")
    logger.info("=======================================================\n")
    
    while True:
        try:
            usr_input = input("[USER]: ")
            if usr_input.strip().lower() in ['quit', 'exit']:
                break
            if not usr_input.strip():
                continue
            
            res = generate_func(usr_input)
            if hasattr(args, "compare") and args.compare and isinstance(res, tuple):
                resp_ft, resp_base = res
                logger.info(f"\n[BASE MODEL]: {resp_base}")
                logger.info(f"[FINETUNED]: {resp_ft}\n")
                log_test_result(usr_input, resp_ft, resp_base)
            else:
                logger.info(f"[LLaMA-2]: {res}\n")
                log_test_result(usr_input, res)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.info(f"\n[ERROR] Inference iteration failed: {e}\n")

if __name__ == "__main__":
    main()
