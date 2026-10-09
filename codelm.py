"""CodeLM (fm.code.generate_small_v2) — PyTorch reconstruction.

Rebuilt from the Apple MIL bytecode (model_0.mpsgraph) in
com_apple_MobileAsset_UAF_FM_CodeLM. Architecture recovered from the op graph:

- 56 layers, hidden 2048, SwiGLU MLP (intermediate 6656), RMSNorm (2/layer +
  final), Qwen3-style q/k RMSNorm applied AFTER RoPE.
- 16 Q heads / 2 KV heads (GQA, head_dim 128); attention scale 1/sqrt(128).
- Layers 0-34: packed QKV projection (2560x2048 = 16Q+2K+2V heads) and their
  own KV cache slots. Layers 35-55: Q-only projection and they ATTEND OVER
  LAYER 34's K/V (cross-layer KV sharing).
- RoPE: interleaved (GPT-J style) with a precomputed (2048, 128) fp16 table
  laid out [sin(64) | cos(64)] per position; max context 2048.
- Embedding: fp16 (153600, 2048); lm_head: separate int4-quantized copy of the
  same weights (tied), dequantized group-32 signed-int4 with fp16 scales.
- fp16 compute; RMSNorm statistics in fp32, result cast back to fp16 before
  multiplying (matches the graph's cast ops). eps: hidden norms 1.0014e-5,
  qk norms 1.0133e-6, final norm 0.

Weights: model.safetensors (fp16) produced by scratch/extract.py from
resources.bin; provenance in weight_map.json.
"""
import argparse
import pathlib

import torch
from safetensors.torch import load_file

HIDDEN = 2048
INTER = 6656
N_LAYER = 56
N_QHEAD = 16
N_KVHEAD = 2
HEAD_DIM = 128
GROUP = 8  # N_QHEAD // N_KVHEAD
VOCAB = 153600
MAX_CTX = 2048
ATTN_SCALE = 0.08837890625  # fp16(1/sqrt(128)) exactly as in the graph
EPS_HIDDEN = 1.0014e-5      # fp16 denormal 0x00a8
EPS_QK = 1.0133e-6          # fp16 denormal 0x0011
EPS_FINAL = 0.0


def rms_norm(x, weight, eps):
    # graph: cast x -> f32, square, mean, +eps(f32), rsqrt, cast -> f16, mul, mul(weight)
    x32 = x.float()
    scale = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * scale.to(x.dtype) * weight


class CodeLM:
    def __init__(self, weights_path, device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        raw = load_file(str(weights_path))
        self.w = {k: v.to(self.device) for k, v in raw.items()}
        self.rope = self.w["rope_table"]  # (2048, 128) fp16
        # per-layer KV cache slots for layers 0..34
        self.cache_k = [torch.zeros(1, N_KVHEAD, MAX_CTX, HEAD_DIM, dtype=torch.float16, device=self.device)
                        for _ in range(35)]
        self.cache_v = [torch.zeros(1, N_KVHEAD, MAX_CTX, HEAD_DIM, dtype=torch.float16, device=self.device)
                        for _ in range(35)]

    def rope_apply(self, x, positions):
        # x: (1, heads, T, 128); interleaved pairs
        t = self.rope[positions]  # (T, 128)
        sin = t[:, :64].to(x.dtype)  # (T, 64)
        cos = t[:, 64:].to(x.dtype)
        sin = sin.view(1, 1, *sin.shape)
        cos = cos.view(1, 1, *cos.shape)
        even = x[..., 0::2]
        odd = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = even * cos - odd * sin
        out[..., 1::2] = odd * cos + even * sin
        return out

    def _split_heads(self, x, n):
        # (1, T, n*128) -> (1, n, T, 128)
        return x.view(x.shape[0], x.shape[1], n, HEAD_DIM).permute(0, 2, 1, 3)

    def forward(self, tokens, past_len):
        """tokens: (1, T) int64. Returns logits (1, T, VOCAB) f16."""
        T = tokens.shape[1]
        pos = torch.arange(past_len, past_len + T, device=self.device)
        x = self.w["model.embed_tokens.weight"][tokens]  # (1, T, 2048) f16
        causal = (pos.view(-1, 1) >= torch.arange(past_len + T, device=self.device).view(1, -1))
        mask = torch.zeros(T, past_len + T, dtype=torch.float16, device=self.device)
        mask.masked_fill_(~causal, float("-inf"))

        for L in range(N_LAYER):
            pre = f"model.layers.{L}"
            h = rms_norm(x, self.w[f"{pre}.input_layernorm.weight"], EPS_HIDDEN)
            if L < 35:
                qkv = torch.matmul(h, self.w[f"{pre}.self_attn.qkv_proj.weight"].t())
                qkv = self._split_heads(qkv, 20)  # (1, 20, T, 128)
                q, k, v = qkv[:, :16], qkv[:, 16:18], qkv[:, 18:20]
                q = self.rope_apply(q, pos)
                k = self.rope_apply(k, pos)
                k = rms_norm(k, self.w[f"{pre}.self_attn.k_norm.weight"].view(1, 1, 1, -1), EPS_QK)
                q = rms_norm(q, self.w[f"{pre}.self_attn.q_norm.weight"].view(1, 1, 1, -1), EPS_QK)
                self.cache_k[L][:, :, past_len:past_len + T] = k
                self.cache_v[L][:, :, past_len:past_len + T] = v
                K, V = self.cache_k[L], self.cache_v[L]
            else:
                q = torch.matmul(h, self.w[f"{pre}.self_attn.q_proj.weight"].t())
                q = self._split_heads(q, 16)
                q = self.rope_apply(q, pos)
                q = rms_norm(q, self.w[f"{pre}.self_attn.q_norm.weight"].view(1, 1, 1, -1), EPS_QK)
                K, V = self.cache_k[34], self.cache_v[34]  # shared with layer 34

            S = past_len + T
            K = K[:, :, :S, :]
            V = V[:, :, :S, :]
            # GQA: tile each KV head 8x (heads 0-7 <- kv0, 8-15 <- kv1)
            K = K.unsqueeze(2).expand(1, N_KVHEAD, GROUP, S, HEAD_DIM).reshape(1, N_QHEAD, S, HEAD_DIM)
            V = V.unsqueeze(2).expand(1, N_KVHEAD, GROUP, S, HEAD_DIM).reshape(1, N_QHEAD, S, HEAD_DIM)

            scores = torch.matmul(q, K.transpose(-1, -2)) * ATTN_SCALE
            scores = scores + mask.view(1, 1, T, S)
            attn = torch.softmax(scores.float(), dim=-1).to(torch.float16)
            out = torch.matmul(attn, V)  # (1, 16, T, 128)
            out = out.permute(0, 2, 1, 3).reshape(1, T, HIDDEN)
            x = x + torch.matmul(out, self.w[f"{pre}.self_attn.o_proj.weight"].t())

            h = rms_norm(x, self.w[f"{pre}.post_attention_layernorm.weight"], EPS_HIDDEN)
            gate = torch.matmul(h, self.w[f"{pre}.mlp.gate_proj.weight"].t())
            up = torch.matmul(h, self.w[f"{pre}.mlp.up_proj.weight"].t())
            act = torch.nn.functional.silu(gate) * up  # graph: mps.swish(gate) * up
            x = x + torch.matmul(act, self.w[f"{pre}.mlp.down_proj.weight"].t())

        x = rms_norm(x, self.w["model.norm.weight"], EPS_FINAL)
        return torch.matmul(x, self.w["lm_head.weight"].t())  # (1, T, 153600)

    @torch.no_grad()
    def generate(self, tokens, max_new=64, stop_id=1):
        toks = list(tokens)
        logits = self.forward(torch.tensor([toks], device=self.device), 0)
        out = []
        for i in range(max_new):
            nxt = int(logits[0, -1].argmax())
            if nxt == stop_id:
                break
            out.append(nxt)
            toks.append(nxt)
            if i < max_new - 1:
                logits = self.forward(torch.tensor([[nxt]], device=self.device), len(toks) - 1)
        return toks, out


def main():
    import sentencepiece as spm

    here = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=str(here / "99bc2924657f80c3f14d5aa10ca392e08bbb2930.asset/AssetData/tokenizer"))
    model = CodeLM(here / "model.safetensors")
    print(f"device: {model.device}", flush=True)

    if args.prompt is None:
        prompt = "<||pre||>```python<n>def add(a, b):<||suf||>    return a + b<n>```<||mid||>"
    else:
        prompt = args.prompt

    toks = special_fix(sp, prompt)
    print(f"prompt tokens: {toks}", flush=True)
    print("decoded prompt:", repr(sp.decode(toks)), flush=True)

    toks, out = model.generate(toks, max_new=args.max_new)
    print("generated ids:", out, flush=True)
    print("GENERATION:", repr(sp.decode(out)), flush=True)


def special_fix(sp, prompt):
    """Encode while keeping <||pre||>/<||suf||>/<||mid||> atomic."""
    specials = [("<||pre||>", 150002), ("<||suf||>", 150003), ("<||mid||>", 150004)]
    out = []
    rest = prompt
    while rest:
        best = None
        for s, sid in specials:
            idx = rest.find(s)
            if idx >= 0 and (best is None or idx < best[0]):
                best = (idx, s, sid)
        if best is None:
            out.extend(sp.encode(rest))
            break
        idx, s, sid = best
        if idx:
            out.extend(sp.encode(rest[:idx]))
        out.append(sid)
        rest = rest[idx + len(s):]
    return out


if __name__ == "__main__":
    main()
