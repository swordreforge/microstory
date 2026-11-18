"""Export the local MacroStories checkpoint to ONNX (dynamic batch / sequence)."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent

HID = 32
INTER = 48
NH = 2
NKV = 1
HD = 16
VOCAB = 378
STEPS = 4
MIX = 0.5
EPS = 1e-6
THETA = 10000.0


class RMS(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(HID))

    def forward(self, x):
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        return (self.weight * (xf * torch.rsqrt(var + EPS))).to(x.dtype)


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(HID, NH * HD, bias=False)
        self.k = nn.Linear(HID, NKV * HD, bias=False)
        self.v = nn.Linear(HID, NKV * HD, bias=False)
        self.o = nn.Linear(NH * HD, HID, bias=False)

    def forward(self, x, cos, sin, mask, first_v):
        B, S, _ = x.shape
        q = self.q(x).view(B, S, NH, HD).transpose(1, 2)
        k = self.k(x).view(B, S, NKV, HD).transpose(1, 2)
        v = self.v(x).view(B, S, NKV, HD).transpose(1, 2)
        if first_v is None:
            first_v = v
        else:
            v = (1.0 - MIX) * v + MIX * first_v

        def rot(t):
            x1, x2 = t[..., : t.shape[-1] // 2], t[..., t.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q = q * cos.unsqueeze(1) + rot(q) * sin.unsqueeze(1)
        k = k * cos.unsqueeze(1) + rot(k) * sin.unsqueeze(1)
        k = k.expand(B, NH, S, HD)
        v = v.expand(B, NH, S, HD)
        att = (q @ k.transpose(-1, -2)) * (HD ** -0.5) + mask
        att = torch.softmax(att.float(), dim=-1).to(q.dtype)
        out = (att @ v).transpose(1, 2).reshape(B, S, NH * HD)
        return self.o(out), first_v


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attn()
        self.gate = nn.Linear(HID, INTER, bias=False)
        self.up = nn.Linear(HID, INTER, bias=False)
        self.down = nn.Linear(INTER, HID, bias=False)
        self.ln1 = RMS()
        self.ln1b = RMS()
        self.ln2 = RMS()
        self.ln2b = RMS()

    def forward(self, x, cos, sin, mask, first_v):
        r = x
        a, first_v = self.attn(self.ln1(x), cos, sin, mask, first_v)
        x = r + self.ln1b(a)
        r = x
        z = self.ln2(x)
        h = self.down(torch.nn.functional.silu(self.gate(z)) * self.up(z))
        return r + self.ln2b(h), first_v


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HID)
        self.layer = Layer()
        self.norm = RMS()
        self.lm_head = nn.Linear(HID, VOCAB, bias=False)
        inv = 1.0 / (THETA ** (torch.arange(0, HD, 2).float() / HD))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, input_ids):
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).float()
        freqs = pos.view(S, 1) * self.inv_freq.view(1, -1)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).expand(B, S, HD)
        sin = emb.sin().unsqueeze(0).expand(B, S, HD)
        mask = torch.full((S, S), torch.finfo(torch.float32).min)
        mask = torch.triu(mask, diagonal=1).view(1, 1, S, S)
        x = self.embed(input_ids)
        first_v = None
        for _ in range(STEPS):
            x, first_v = self.layer(x, cos, sin, mask, first_v)
            x = self.norm(x)
        return self.lm_head(x)


def load_model(weights: Path = ROOT / "model.safetensors") -> Model:
    from safetensors.torch import load_file

    sd = load_file(str(weights))
    p = "model.layers.0."
    mapping = {
        "embed.weight": "model.embed_tokens.weight",
        "layer.attn.q.weight": p + "self_attn.q_proj.weight",
        "layer.attn.k.weight": p + "self_attn.k_proj.weight",
        "layer.attn.v.weight": p + "self_attn.v_proj.weight",
        "layer.attn.o.weight": p + "self_attn.o_proj.weight",
        "layer.gate.weight": p + "mlp.gate_proj.weight",
        "layer.up.weight": p + "mlp.up_proj.weight",
        "layer.down.weight": p + "mlp.down_proj.weight",
        "layer.ln1.weight": p + "input_layernorm.weight",
        "layer.ln1b.weight": p + "input_layernorm_2.weight",
        "layer.ln2.weight": p + "post_attention_layernorm.weight",
        "layer.ln2b.weight": p + "post_attention_layernorm_2.weight",
        "norm.weight": "model.norm.weight",
        "lm_head.weight": "model.embed_tokens.weight",
    }
    model = Model()
    model.load_state_dict({k: sd[v] for k, v in mapping.items()}, strict=True)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "converted" / "macro_stories.onnx"))
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = load_model()
    dummy = torch.randint(0, VOCAB, (1, 8))
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
