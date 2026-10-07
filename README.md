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

MacroStories is a **19,969-parameter autoregressive Transformer** that generates complete English children's stories from a single start token. Her FP32 weights occupy just **81,364 bytes—smaller than many PNG images**.

[TinyStories](https://arxiv.org/abs/2305.07759) showed that small Transformers can write coherent English when trained on a suitable distribution of synthetic stories. 

We takes this question down to about **20 thousand parameters**, roughly one fiftieth of a one-million-parameter model and proved that a model even this small can write a story of 100–300 words that establishes a goal, takes relevant actions, and reaches an outcome that resolves it.

## Quick start

```bash
pip install "transformers==4.55.0" "safetensors>=0.4.3"
```

```python
import runpy

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "raincandy-u/MacroStories"

tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(
    repo_id, trust_remote_code=True
).eval()

decode_story = runpy.run_path(
    hf_hub_download(repo_id, "render_story.py")
)["decode_story"]

torch.manual_seed(20261007)
prompt = torch.tensor([[tokenizer.bos_token_id]])

with torch.inference_mode():
    output = model.generate(
        input_ids=prompt,
        attention_mask=torch.ones_like(prompt),
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
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

The display helper maps numbered character tokens to `Alex`, `Robin`, and `Casey`. `raw_story` retains the original character markers.

## Architecture

One decoder block is applied **four times with shared weights**. The input embedding and output projection also share a weight matrix, which accounts for **60.6% of all parameters**.

| Component | Configuration |
|---|---|
| Vocabulary | 378 WordLevel tokens |
| Hidden dimension | 32 |
| SwiGLU intermediate dimension | 48 |
| Decoder blocks | 1, applied 4 times |
| Attention | 2 query heads, 1 KV head; head dimension 16 |
| Position encoding | RoPE |
| Normalization | RMSNorm before and after attention and FFN sublayers, plus final output RMSNorm |
| Value residual | Passes 2–4 mix current value vectors equally with those from pass 1 |
| Input embedding / output projection | Shared weight matrix |

| Parameter group | Parameters |
|---|---:|
| Shared embedding / output projection | 12,096 |
| Attention projections | 3,072 |
| SwiGLU projections | 4,608 |
| RMSNorm | 160 |
| Inactive early-exit gate | 33 |
| **Total** | **19,969** |

MacroStories was trained from random initialization using an implementation adapted from [ByteDance/Ouro](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1).

## Training

We gave Codex a goal and guidance when needed. She autonomously carried out the entire workflow, generating data, forming hypotheses, testing them, refining them from the results, and testing again—through **38 experimental rounds** to final model selection.

Stories were generated with a local Gemma model served through vLLM, using vocabulary constraints and numbered character markers. Codex reviewed complete stories for grammatical correctness, goal resolution, and character and object consistency.

Training used next-token cross-entropy on the fourth recurrent pass. Muon updated the decoder's two-dimensional linear weight matrices; AdamW updated the remaining parameters.

| Setting | Value |
|---|---|
| Training updates | 4,000 |
| Batch size | 32 |
| Learning rate | 0.003 |
| Learning-rate schedule | Cosine decay with 5% warmup |
| Training seed | 456 |
| Hardware | One RTX 3090 |

## Evaluation

| Criterion | Stories meeting the criterion |
|---|---:|
| Goal resolved through relevant actions and an outcome | 98 / 100 |
| No grammatical error on a full reading | 80 / 100 |
| Consistent characters and objects | 91 / 100 |
| Length of 100–300 words | 82 / 100 |
| **All four criteria together** | **65 / 100** |

## Example output

> "I want to help this flower," Alex said. The flower was dry in the village garden. Alex wanted to water the flower. Alex grabbed a large bucket, but the bucket was too heavy for them to lift. Alex looked at the small cup on the bench. Alex dipped the cup into the bucket. Alex walked to the flower and poured the water onto the soil. Alex went back and forth. Alex made several trips with the small cup. Alex poured more water into the dirt. Alex and Robin watched the plant. Finally, the flower got enough water. Alex smiled because the flower was not dry now.

## Capabilities

The model is designed only to generate English children's stories. She has **no world knowledge or instruction-following capability**. She does not support factual question answering, conversation, or writing on arbitrary topics.

## Acknowledgments

We thank the authors of [TinyStories](https://arxiv.org/abs/2305.07759) for showing how carefully designed synthetic data enables coherent story generation in small language models, and the [ByteDance/Ouro](https://huggingface.co/ByteDance/Ouro-1.4B) team for their work on shared-weight recurrent Transformers and the implementation that served as the foundation for MacroStories.

We also thank the Hugging Face team and community for the libraries, tools, and open ecosystem that made this project possible—from training on a single consumer GPU to sharing the finished model with everyone.