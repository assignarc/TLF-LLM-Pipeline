---
license: apache-2.0
base_model: sarvamai/sarvam-1
library_name: peft
tags:
- indic-nlp
- dictionary
- sanskrit
- marathi
- hindi
- TransLiteral
- Kannada
- Oriya
- Indic
- Punjabi
- sft
- lora
---
# TLF-7B-LLM-01

## Model Description

This model is a fine-tuned version of [sarvamai/sarvam-1](https://huggingface.co/sarvamai/sarvam-1) specialized for **Bilingual Indic Lexicography**.

It has been trained to provide structured morphological breakdowns, definitions, and regional translations for Sanskrit and other Indian regional languages.

The training data was ingested through the **TLF Mega-Pipeline**, integrating structured dictionary databases (MSSQL) with unstructured regional texts to improve grammar and stylistic intelligence.

### Data Source :

The dictionary content is freely available as Unified Dictionary project on [TransLiteral Foundation&#39;s website](https://www.transliteral.org/dictionary/). The website provides 1,153,927 Words and their 2,309,309 Meanings from 71 [dictionaries](https://www.transliteral.org/dictionary/all.kosh/source). These are cited with over 1079 [literary sources](https://www.transliteral.org/dictionary/all.references/text) from several authors from ancient Indian regional and religious texts. The source is used under [Creative Commons - ShareALike International License. ](https://creativecommons.org/licenses/by-nc-sa/4.0/)

### Intended Use

- **Dictionary Lookups**: Providing high-accuracy definitions and etymologies.
- **Morphological Analysis**: Breaking down complex Sanskrit/Indic root words.
- **Regional Translation**: Translating word concepts across Marathi, Hindi, and English.

## Training Hyperparameters

The following hyperparameters were used during training:

- **Engine**: MLX
- **Learning Rate**: 2e-05
- **Batch Size**: 1
- **Gradient Accumulation**: 64
- **Optimizer**: adamw_torch
- **LR Scheduler**: cosine
- **LoRA R**: 32
- **LoRA Alpha**: 16
- **Max Sequence Length**: 1024

## Prompt Template

To achieve the intended structured output, use the following prompt format:

```text
<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{query} [/INST]
```

## Inference Example

### Using MLX (Apple Silicon)

```python
import mlx_lm
model, tokenizer = mlx_lm.load("AssignArc/TLF-7B-LLM-01")

prompt = "Provide a comprehensive morphological breakdown for: 'Abacus'"
# Use Sarvam/Llama template logic here
response = mlx_lm.generate(model, tokenizer, prompt=prompt)
print(response)
```

### Using Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("sarvamai/sarvam-1")
model = PeftModel.from_pretrained(base_model, "AssignArc/TLF-7B-LLM-01")
tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-1")

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

## Citation & Credits

- **TLF Framework**: Architected for Unified Indic LLM Fine-tuning.
- **Data Source**: Custom Dictionary & Regional Text Corpus.
