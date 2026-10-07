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

**MacroStories is a 19,969-parameter autoregressive Transformer trained from scratch to write complete English children's stories from a single start token.** Its FP32 weight file is **81,364 bytes**. It loads through the standard Hugging Face interfaces with `trust_remote_code=True` and supports CPU inference.

[TinyStories](https://arxiv.org/abs/2305.07759) showed that the parameter threshold for coherent English depends on the training distribution. Models below 10 million parameters, including models with a single Transformer block, wrote fluent stories when trained on synthetic text restricted to a young child's vocabulary. Its smallest models, at roughly one million parameters, still had visible grammatical weaknesses.

MacroStories asks the question one step further down: can a Transformer with about 20 thousand parameters, roughly one fiftieth of a one-million-parameter model, write a story of 100–300 words that establishes a goal, takes relevant actions, and reaches an outcome that resolves it? The model receives only a beginning-of-sequence (BOS) token. Its stories cover six plot families:

- carrying bread home;
- building a birdhouse;
- mending a kite;
- retrieving a ball;
- returning a book;
- watering a flower.

After the final weights were frozen, **98 of 100 sampled stories carried a goal through to an outcome**, and **80 contained no clear grammatical error on a full reading**. **65 met four strict conditions together**: no clear grammatical error, a resolved goal, consistent characters and objects, and a length of 100–300 words. These are AI-assisted assessments within the six controlled plot families, as detailed below.

At this scale, parameter allocation matters. The shared input and output embedding holds 12,096 parameters, **60.6% of the total**; attention and feed-forward layers together hold fewer parameters than the embedding. Changes to the data and its representation affected the outcome more than changes to the architecture and optimizer. We interpret this as evidence that grammar and the state carried across a simple story can be learned within a very small parameter budget when the training distribution is narrow and consistent. This does not establish the capacity required for unrestricted English or open-domain storytelling.

The model was built through an agent-led research process. A human supplied the goal and constraints. The main agent, Codex running GPT-6.1-Sol, worked with GPT-6-Luna subagents through **38 numbered experimental rounds on one RTX 3090**, generated training data with a local Gemma model, and chose subsequent experiments from earlier results. The release contributes a working model at this scale, evidence about the importance of the training distribution, and a record of an agent carrying out a sustained research project with limited human guidance.

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

The display helper consistently replaces numbered character tokens with `Alex`, `Robin`, and `Casey`; it does not rewrite sentences or repair errors. `raw_story` preserves the character-token text. These markers are registered as special tokens, so `skip_special_tokens=True` would remove character references; use the helper to retain them. This example generates one story. The reported evaluation used batch size eight, which changes the sampling random-number sequence even with the same seed.

## Architecture

A single physical decoder block is applied four times with shared weights. Recurrence increases computation without adding four copies of the parameters.

| Setting | Value |
|---|---|
| Unique parameters | 19,969 |
| Tokenizer | WordLevel, 378 tokens |
| Hidden size / FFN intermediate size | 32 / 48 |
| Physical blocks / recurrent passes | 1 / 4 |
| Attention | 2 query heads, 1 KV head, head dimension 16 |
| Activation / position encoding | SwiGLU / RoPE |
| Normalization | RMSNorm before and after attention/FFN branches |
| Value residual mixing coefficient | 0.5 |
| Input/output embeddings | Tied |
| Maximum sequence length | 2,048 tokens |
| Weight dtype / file size | FP32 / 81,364 bytes |

| Parameter group | Parameters |
|---|---:|
| Shared input/output embedding | 12,096 |
| Attention | 3,072 |
| Feed-forward network | 4,608 |
| RMSNorm | 160 |
| Retained exit gate, with early exit disabled | 33 |
| **Total** | **19,969** |

Architecture code is adapted from [ByteDance/Ouro-1.4B at revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1). **MacroStories was initialized randomly and did not load Ouro's pretrained weights.** Existing source license headers are retained.

## Training data and procedure

The final training set contains **1,631 unique synthetic stories**, or **233,756 tokens including BOS/EOS**. Validation and test contain **40 stories each**. The 378-token vocabulary was built from an initial 328 training stories and then frozen; later expansion added training stories without changing the vocabulary or held-out sets.

Stories were generated locally with Gemma4 26B A4B W4A16 served through vLLM: temperature 1.0, top-p 0.95, top-k 64, and thinking budget 16. Teacher generation was constrained to limited surface word forms and character markers. AI agents read complete stories to filter language errors, unresolved goals, and, in the final expansion stage, identity inconsistencies. Additional training stories with words outside the frozen vocabulary were excluded.

Training-family counts are 206 bread, 210 birdhouse, 271 kite, 324 ball, 270 book, and 350 flower stories. Third-person character references use `they/them/their`; dialogue retains `I/you`, and objects retain `it`.

| Training setting | Value |
|---|---|
| Selected run | `r38_20k_lr003_seed456` |
| Objective | Next-token cross-entropy on the final recurrent pass |
| Optimizer | Muon for decoder 2D Linear weights; AdamW for other parameters |
| Learning rate / weight decay | 0.003 / 0.01 |
| Batch size / updates | 32 / 4,000 |
| Schedule / gradient clipping | Cosine decay, 5% warmup / 1.0 |
| Training seed / data seed | 456 / 42 |
| Selected checkpoint | Step 4,000 |
| Training hardware | One RTX 3090 |

Training used synthetic-text supervision without GRPO, DPO, or RLHF. The final model does not use auxiliary recurrent losses, learned loop embeddings, input embedding reinjection, or adaptive early exit. The [final configuration](experiments/final_20k.json), [round-by-round record](reports/20k_rounds.md), and [final report](reports/final_20k_report.md) are included. The training corpus is not bundled in this model repository.

## Evaluation

### Frozen-model story evaluation

One hundred new stories were sampled after model selection and weight freezing: BOS-only input, CPU FP32 inference, batch size 8, seed `20261007`, temperature 0.5, top-p 0.9, `min_new_tokens=0`, and `max_new_tokens=2047`.

Three AI subagents read disjoint randomized subsets of 34, 33, and 33 complete stories with model metadata hidden. Each story was reviewed by one agent; this was not a three-reviewer consensus assessment. These are AI-assisted evaluations rather than human annotations. Gemma scoring was not used as the final label source.

| Criterion | Passing stories / 100 |
|---|---:|
| Readable main storyline, allowing local errors | 98 |
| No clear grammatical error on a full reading | 80 |
| Initial goal resolved by relevant actions and an outcome | 98 |
| Consistent characters, objects, and important states | 91 |
| Length of 100–300 words | 82 |
| All four strict conditions: grammar, goal, consistency, length | **65** |

All 100 reached EOS. None was truncated, and there were no duplicate whole stories. Mean length was **108.19 words**; **18 stories were shorter than 100 words**. Mean repeated four-gram rate was **0.030064**; absence of whole-story duplicates does not imply absence of repetition within stories.

Inspect the [original samples](reports/final100_samples_t05.jsonl), [per-story labels](reports/final100_labels.jsonl), [machine-readable summary](reports/final_20k_summary.json), and review notes ([part 1](reports/final100_part1_review.md), [part 2](reports/final100_part2_review.md), [part 3](reports/final100_part3_review.md)). Results apply to this sample and generation setup rather than general writing accuracy.

### Held-out likelihood

Trainer validation cross-entropy was **0.869492**. A separate evaluation of the 40-story test set produced token-weighted cross-entropy **0.872573 nats** over 5,678 predicted tokens, including EOS and excluding prediction of BOS. These evaluations use different aggregation methods.

The test encoding contains two unknown tokens. Original-text bits per byte is therefore not reported: unknown-token substitutions prevent exact likelihood comparison on the original text. See the [test likelihood record](reports/final_test_likelihood.json).

## Example output and observed failures

Final-evaluation sample 20 has 106 words and passed all four strict conditions. Only the character-name display replacement was applied:

> "I want to help this flower," Alex said. The flower was dry in the village garden. Alex wanted to water the flower. Alex grabbed a large bucket, but the bucket was too heavy for them to lift. Alex looked at the small cup on the bench. Alex dipped the cup into the bucket. Alex walked to the flower and poured the water onto the soil. Alex went back and forth. Alex made several trips with the small cup. Alex poured more water into the dirt. Alex and Robin watched the plant. Finally, the flower got enough water. Alex smiled because the flower was not dry now.

Observed failures include **“The flower got enough water and poured Alex.”** in sample 11 and **“Alex had a good and held the bread inside the blue sky.”** in sample 37. The original samples retain these errors. Outputs may still mix agent/object roles, contain malformed sentences, repeat phrases, or end before 100 words.

## Intended use and limitations

MacroStories is intended for research into very small autoregressive Transformers, controlled story generation, parameter allocation, and synthetic-data representation. It can serve as a compact, inspectable example of a recurrent Transformer released through Hugging Face.

Its tested scope is limited English vocabulary, six controlled children's-story plot families, and autonomous BOS generation. Arbitrary prompts, unrestricted topics, multilingual generation, factual answering, instruction following, and conversation have not been validated. The fixed WordLevel vocabulary cannot represent arbitrary words. The 2,048-token architectural limit does not establish reliable long-context behavior.

The experiments suggest that a narrow, consistent training distribution makes this task possible at about 20 thousand parameters. They do not prove a global minimum parameter count, a globally optimal architecture, or equivalent performance to TinyStories under matched training and evaluation conditions. Earlier larger candidates and different vocabularies are not directly comparable to this final release.

## Reference

Ronen Eldan and Yuanzhi Li. *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* 2023. [arXiv:2305.07759](https://arxiv.org/abs/2305.07759).
