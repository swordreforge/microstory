---
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- custom-code
- storytelling
- small-language-model
- recurrent-depth
- synthetic-data
---

# MacroStories

## How small can an autoregressive Transformer be and still write a complete story?

**MacroStories is a ~20k-parameter autoregressive Transformer trained from scratch to write complete English children's stories.** Its FP32 weight file is only **81,364 bytes**.

[TinyStories](https://arxiv.org/abs/2305.07759) showed that the parameter threshold for coherent English depends on the training distribution. Models below 10 million parameters, including models with a single Transformer block, wrote fluent stories when trained on synthetic text restricted to a young child's vocabulary. 

MacroStories asks the question one step further down: can a Transformer with about 20 thousand parameters, roughly one fiftieth of a one-million-parameter model, write a story of 100–300 words that establishes a goal, takes relevant actions, and reaches an outcome that resolves it? The answer is yes.

The model was built through an agent-led research process. A human supplied the goal and constraints. The main agent, Codex running GPT-6.1-Sol, worked with GPT-6-Luna subagents through **38 numbered experimental rounds on one RTX 3090**, generated training data with a local Gemma model, and chose subsequent experiments from earlier results.

## Quick start

Install PyTorch for your platform, then the inference dependencies:

```bash
pip install "transformers==4.55.0" "safetensors>=0.4.3"
```

The original project export was tested with Python 3.13, PyTorch 2.13, and Transformers 4.55.0. Custom architecture code is included in this repository and requires `trust_remote_code=True`.

```python
import runpy

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "raincandy-u/MacroStories"
tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo_id, trust_remote_code=True
).eval()
decode_story = runpy.run_path(
    hf_hub_download(repo_id, "render_story.py")
)["decode_story"]

torch.set_num_threads(4)
torch.manual_seed(20261007)
prompt = torch.tensor([[tokenizer.bos_token_id]])
with torch.inference_mode():
    output = model.generate(
        input_ids=prompt,
        attention_mask=torch.ones_like(prompt),
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        min_new_tokens=0,
        max_new_tokens=2047,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
raw_story, story, names = decode_story(
    output[0, 1:].tolist(), tokenizer, [tokenizer.eos_token_id]
)
print(story)
```

The decoder consistently replaces numbered character tokens with `Alex`, `Robin`. `raw_story` preserves the character-token text.

These markers are registered as special tokens, so `skip_special_tokens=True` would remove character references;

## Architecture

MacroStories is a decoder-only autoregressive Transformer with **19,969 parameters**. It applies one decoder block four times, sharing the block’s weights across all four passes. The input embedding and output projection also share weights.

| Component | Configuration |
|---|---|
| Vocabulary | 378 WordLevel tokens |
| Hidden dimension | 32 |
| SwiGLU intermediate dimension | 48 |
| Decoder blocks | 1, applied 4 times |
| Attention | 2 query heads and 1 shared key/value head; head dimension 16 |
| Position encoding | RoPE |
| Normalization | RMSNorm before and after each attention and feed-forward sublayer, plus a final output RMSNorm |
| Value residual | In passes 2–4, average the current value projection with the first pass’s value projection |
| Input/Output embedding | Shared weight matrix |

| Parameter group | Parameters |
|---|---:|
| Shared embedding / output projection | 12,096 |
| Attention projections | 3,072 |
| SwiGLU projections | 4,608 |
| RMSNorm | 160 |
| Inactive early-exit gate retained in the checkpoint | 33 |
| **Total** | **19,969** |

The implementation is adapted from [ByteDance/Ouro](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1).

MacroStories was trained from random initialization; no pretrained Ouro weights were used.

## Training data and procedure

Stories were generated locally with Gemma-4 served through vLLM. Teacher generation was constrained to limited surface word forms and character markers. AI agents read complete stories to filter language errors, unresolved goals, and, in the final expansion stage, identity inconsistencies. Additional training stories with words outside the frozen vocabulary were excluded.

## Example output

> "I want to help this flower," Alex said. The flower was dry in the village garden. Alex wanted to water the flower. Alex grabbed a large bucket, but the bucket was too heavy for them to lift. Alex looked at the small cup on the bench. Alex dipped the cup into the bucket. Alex walked to the flower and poured the water onto the soil. Alex went back and forth. Alex made several trips with the small cup. Alex poured more water into the dirt. Alex and Robin watched the plant. Finally, the flower got enough water. Alex smiled because the flower was not dry now.

## Intended use and limitations

MacroStories is intended for research, and its tested scope is limited English vocabulary. It has **no world knowledge or instruction-following capability**.

It cannot answer factual questions, follow user requests, hold a conversation, or write about arbitrary topics. Even within this scope, outputs can contain grammatical errors, inconsistent character or object references, repeated phrases, and stories shorter than the target length.

## Reference

Ronen Eldan and Yuanzhi Li. *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* 2023. [arXiv:2305.07759](https://arxiv.org/abs/2305.07759).
