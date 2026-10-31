"""Drop a declared-but-missing multi-token-prediction (nextn) layer from a Qwen3.5 GGUF.

TRL/transformers save the trained model without Qwen3.5's MTP weights, but llama.cpp's converter still
declares them (block_count = layers + 1, nextn_predict_layers = 1), so Ollama fails with
"missing tensor 'blk.<n>.attn_norm.weight'". MTP is unused for normal decoding; this rewrites only the
header (block_count, nextn_predict_layers, per-layer arrays) and copies the tensor data unchanged.

Usage: python scripts/gguf_drop_nextn.py in.gguf out.gguf
"""
import shutil, struct, sys

FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def main(src, dst):
    f = open(src, "rb")
    magic, ver, n_t, n_kv = struct.unpack("<4sIQQ", f.read(24))
    assert magic == b"GGUF" and ver == 3, (magic, ver)

    def rstr():
        n, = struct.unpack("<Q", f.read(8)); return f.read(n)

    def rval(t):
        if t == 8:
            return rstr()
        if t == 9:
            et, n = struct.unpack("<IQ", f.read(12)); return (et, [rval(et) for _ in range(n)])
        return struct.unpack("<" + FMT[t], f.read(struct.calcsize(FMT[t])))[0]

    def wval(t, v):
        if t == 8:
            return struct.pack("<Q", len(v)) + v
        if t == 9:
            et, xs = v; return struct.pack("<IQ", et, len(xs)) + b"".join(wval(et, x) for x in xs)
        return struct.pack("<" + FMT[t], v)

    kvs = []
    for _ in range(n_kv):
        k = rstr().decode(); t, = struct.unpack("<I", f.read(4)); kvs.append([k, t, rval(t)])
    tinfo_start = f.tell()
    names = []
    for _ in range(n_t):
        n = rstr(); nd, = struct.unpack("<I", f.read(4)); f.read(8 * nd + 4 + 8); names.append(n.decode())
    tinfo = open(src, "rb").read()[tinfo_start:f.tell()]
    kv = {k: (t, v) for k, t, v in kvs}
    arch = kv["general.architecture"][1].decode()
    align = kv.get("general.alignment", (4, 32))[1]
    data_start = -(-f.tell() // align) * align

    n_blocks = kv[f"{arch}.block_count"][1]
    present = len({n.split(".")[1] for n in names if n.startswith("blk.")})
    drop = n_blocks - present
    assert drop == kv.get(f"{arch}.nextn_predict_layers", (4, 0))[1] and drop > 0, \
        f"block_count {n_blocks}, blocks present {present}: nothing to drop or not an MTP mismatch"
    for e in kvs:
        if e[0] == f"{arch}.block_count":
            e[2] = present
        elif e[0] == f"{arch}.nextn_predict_layers":
            e[2] = 0
        elif e[1] == 9 and len(e[2][1]) == n_blocks:  # per-layer arrays lose the dropped entries
            e[2] = (e[2][0], e[2][1][:present])

    head = struct.pack("<4sIQQ", magic, ver, n_t, n_kv)
    head += b"".join(wval(8, k.encode()) + struct.pack("<I", t) + wval(t, v) for k, t, v in kvs) + tinfo
    with open(dst, "wb") as o:
        o.write(head + b"\0" * (-len(head) % align))
        f.seek(data_start); shutil.copyfileobj(f, o, 64 << 20)
    print(f"{arch}: block_count {n_blocks} -> {present}, nextn 0, {n_t} tensors copied to {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
