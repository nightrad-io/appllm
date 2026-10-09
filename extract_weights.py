"""Extract CodeLM weights from resources.bin into model.safetensors.

Walks the parsed MIL op graph (/tmp/mil_full.json, produced by final.py) to map
every resource to a named model tensor, dequantizes int4 group-32 weights
(signed, low-nibble first, fp16 scale per (row, group of 32 along input dim)),
and writes fp16 safetensors plus weight_map.json.

Model layout (verified against op walk):
- 56 layers; hidden 2048; 16 Q heads / 2 KV heads; head_dim 128; MLP 6656.
- Layers 0-34: packed QKV (2560,2048), own K/V cache slots, q_norm + k_norm.
- Layers 35-55: Q-only proj (2048,2048), reuse layer 34's K/V, q_norm only.
- Norms: input/post_attention per layer + final (113); qk norms (91).
- Embedding fp16 (153600,2048); lm_head = int4-quantized tied copy (corr .993).
- RoPE: precomputed (2048,128) fp16 table = [sin(64) | cos(64)] per position.

Prereq: python3 final.py <asset>/.../model_0.mpsgraph   (writes /tmp/mil_full.json)
"""
import json, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE
BASE = ROOT / "54e85bb601a5f3e2a0e85b71a616c4c5fe462893.asset/.AssetData/model.bundle/H13.bundle/main/main_mps_graph/main_mps_graph.mpsgraphpackage"

d = json.load(open("/tmp/mil_full.json"))
ops, attrs, props, strings = d["ops"], d["attrs"], d["props"], d["strings"]
import plistlib
man = plistlib.load(open(BASE / "manifest.plist", "rb"))
offs = {k: int(v) for k, v in man["Package Version"]["6.3.4"]["Resource Offsets"].items()}
rep = json.load(open(ROOT / "arch_report.json"))
res = {r["name"]: r for r in rep["resources"]}


def varint(b, o):
    c = b[o]
    if c & 1:
        return c >> 1, o + 1
    if c == 0:
        return -1, o + 9
    n = 0
    x = c
    while x & 1 == 0:
        x >>= 1
        n += 1
    return int.from_bytes(b[o:o + n + 1], "little") >> (n + 1), o + n + 1


def const_resource(op):
    v, _ = varint(bytes.fromhex(props[op["props"]]), 0)
    if 4374 <= v < 5142:
        b = bytes.fromhex(attrs[v]["h"])
        vals, o = [], 0
        while o < len(b):
            x, o = varint(b, o)
            vals.append(x)
        return strings[44 + vals[2]]
    return None  # inline constant


val2pos = {}
vid = 5
for pos, op in enumerate(ops):
    for k in range(op["nres"]):
        val2pos[vid + k] = pos
    vid += op["nres"]

RES = {}
for pos, op in enumerate(ops):
    if op["op"] == "mps.constant":
        r = const_resource(op)
        if r:
            RES[pos] = r

dqs, ln2048, ln128 = [], [], []
embed_res, rope_res = None, None

for pos, op in enumerate(ops):
    name = op["op"]
    if name == "mps.dequantize":
        ins = [RES.get(val2pos[vi]) for vi in op["in"]]
        dqs.append(tuple(ins))
    elif name == "mps.multiply" and len(op["in"]) == 2:
        for vi in op["in"]:
            src = val2pos.get(vi)
            if src is not None and src in RES:
                shp = res[RES[src]]["shape"][0]
                if shp == [4096]:      # encoded dims: true (2048,)
                    ln2048.append(RES[src])
                elif shp == [256]:     # true (128,)
                    ln128.append(RES[src])
    elif name == "mps.gather" and len(op["in"]) == 3:
        src = val2pos.get(op["in"][0])
        if src is not None and src in RES:
            shp = res[RES[src]]["shape"][0]
            if shp == [307200, 4096]:
                embed_res = RES[src]
            elif shp == [4096, 256]:
                rope_res = RES[src]

dqs = [q for q in dqs if q[0] and q[1]]
assert len(dqs) == 281, f"dq count {len(dqs)}"
assert len(ln2048) == 113, f"ln2048 {len(ln2048)}"
assert len(ln128) == 91, f"ln128 {len(ln128)}"
assert embed_res == "tmp" and rope_res == "tmp_1", (embed_res, rope_res)

SCALE_OF = {q[0]: q[1] for q in dqs}

f = open(BASE / "resources.bin", "rb")


def read_array(name):
    r = res[name]
    f.seek(offs[name])
    b0 = f.read(1)[0]
    nbytes = b0.bit_length() - 1          # 02->1 04->2 08->3 10->4
    hdr = f.read(nbytes + 1)              # nbytes length bytes + type byte
    datalen = int.from_bytes(hdr[:nbytes], "little") << (7 - nbytes)
    assert datalen == r["size"] - (nbytes + 2), (name, datalen, r["size"])
    shape = [x // 2 for x in r["shape"][0]]
    elem = r["shape"][1]
    data = f.read(datalen)
    if elem == 39:                        # fp16
        assert len(data) == 2 * int(np.prod(shape))
        return np.frombuffer(data, dtype="<f2").reshape(shape)
    if elem == 53:                        # int4 packed 2/byte
        assert len(data) == int(np.prod(shape)) // 2
        w = np.frombuffer(data, dtype=np.uint8).reshape(shape[0], shape[1] // 2)
        lo = (w & 0xF).astype(np.int8)
        hi = (w >> 4).astype(np.int8)
        v = np.empty((w.shape[0], w.shape[1] * 2), dtype=np.int8)
        v[:, 0::2] = lo                   # low nibble = even element index
        v[:, 1::2] = hi
        v = np.where(v > 7, v - 16, v)    # signed int4
        v = v.reshape(shape[0], shape[1] // 32, 32)
        s = read_array(SCALE_OF[name]).astype(np.float32)
        assert s.shape == (shape[0], shape[1] // 32), (name, s.shape)
        return (v.astype(np.float32) * s[:, :, None]).reshape(shape).astype(np.float16)
    raise ValueError(f"elem {elem} for {name}")


tensors, wmap = {}, {}


def put(tname, arr, src):
    tensors[tname] = np.ascontiguousarray(arr)
    wmap[tname] = src


put("model.embed_tokens.weight", read_array("tmp"), "tmp")
put("rope_table", read_array("tmp_1"), "tmp_1")
put("model.norm.weight", read_array(ln2048[112]), ln2048[112])

head_w, head_s = dqs[280][0], dqs[280][1]
assert res[head_w]["shape"][0] == [307200, 4096]
put("lm_head.weight", read_array(head_w), f"{head_w}+{head_s}")

for L in range(56):
    pre = f"model.layers.{L}"
    qkv_or_q = dqs[5 * L]
    o, gate, up, down = dqs[5 * L + 1][0], dqs[5 * L + 2][0], dqs[5 * L + 3][0], dqs[5 * L + 4][0]
    proj = "self_attn.qkv_proj" if L < 35 else "self_attn.q_proj"
    put(f"{pre}.{proj}.weight", read_array(qkv_or_q[0]), f"{qkv_or_q[0]}+{qkv_or_q[1]}")
    put(f"{pre}.self_attn.o_proj.weight", read_array(o), f"{o}+{SCALE_OF[o]}")
    put(f"{pre}.mlp.gate_proj.weight", read_array(gate), f"{gate}+{SCALE_OF[gate]}")
    put(f"{pre}.mlp.up_proj.weight", read_array(up), f"{up}+{SCALE_OF[up]}")
    put(f"{pre}.mlp.down_proj.weight", read_array(down), f"{down}+{SCALE_OF[down]}")
    put(f"{pre}.input_layernorm.weight", read_array(ln2048[2 * L]), ln2048[2 * L])
    put(f"{pre}.post_attention_layernorm.weight", read_array(ln2048[2 * L + 1]), ln2048[2 * L + 1])
    if L < 35:
        put(f"{pre}.self_attn.k_norm.weight", read_array(ln128[2 * L]), ln128[2 * L])
        put(f"{pre}.self_attn.q_norm.weight", read_array(ln128[2 * L + 1]), ln128[2 * L + 1])
    else:
        put(f"{pre}.self_attn.q_norm.weight", read_array(ln128[70 + (L - 35)]), ln128[70 + (L - 35)])

for L in range(56):
    pre = f"model.layers.{L}"
    assert tensors[f"{pre}.self_attn.o_proj.weight"].shape == (2048, 2048)
    assert tensors[f"{pre}.mlp.gate_proj.weight"].shape == (6656, 2048)
    assert tensors[f"{pre}.mlp.up_proj.weight"].shape == (6656, 2048)
    assert tensors[f"{pre}.mlp.down_proj.weight"].shape == (2048, 6656)
    assert tensors[f"{pre}.input_layernorm.weight"].shape == (2048,)
assert tensors["model.embed_tokens.weight"].shape == (153600, 2048)
assert tensors["lm_head.weight"].shape == (153600, 2048)
assert tensors["rope_table"].shape == (2048, 128)

hw = tensors["lm_head.weight"][:2000].astype(np.float64)
ew = tensors["model.embed_tokens.weight"][:2000].astype(np.float64)
print(f"head-vs-embedding corr: {np.corrcoef(hw.ravel(), ew.ravel())[0,1]:.6f}")

import safetensors.numpy as st

out = ROOT / "model.safetensors"
st.save_file(tensors, str(out))
json.dump(wmap, open(ROOT / "weight_map.json", "w"), indent=1)
print(f"wrote {out} ({out.stat().st_size/1e9:.2f} GB), {len(tensors)} tensors")
