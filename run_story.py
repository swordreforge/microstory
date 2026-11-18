"""Generate a children's story from the local MacroStories checkpoint."""

import argparse
import runpy
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    return "xpu" if torch.xpu.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "xpu"])
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--max-new-tokens", type=int, default=2047)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    device = pick_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(ROOT)
    model = (
        AutoModelForCausalLM.from_pretrained(
            ROOT, trust_remote_code=True, torch_dtype=torch.float32
        )
        .eval()
        .to(device)
    )
    decode_story = runpy.run_path(str(ROOT / "render_story.py"))["decode_story"]

    torch.manual_seed(args.seed)
    prompt = torch.tensor([[tokenizer.bos_token_id]], device=device)
    with torch.inference_mode():
        output = model.generate(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    _, story, _ = decode_story(
        output[0, 1:].tolist(), tokenizer, [tokenizer.eos_token_id]
    )
    print(story)


if __name__ == "__main__":
    main()
