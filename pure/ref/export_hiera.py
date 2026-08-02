# Extract the MedSAM2 (SAM2 Hiera-T) IMAGE ENCODER (Hiera trunk + FPN neck) into a forward-order weight
# blob + staged parity refs. B=1, input 1024 -> stage outputs 256²/128²/64²/32² -> neck 4×256². The
# pos_embed (bicubic-interp + window-tile) is precomputed to a [256,256,96] constant (no bicubic in C++).
import os, numpy as np, torch
from sam2.build_sam import build_sam2
HERE = os.path.dirname(__file__)
m = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", os.path.join(HERE, "MedSAM2_latest.pt"), device="cpu")
m.eval(); enc = m.image_encoder; tr = enc.trunk
def P(t): return t.detach().cpu().numpy()
blob = bytearray()
def put(a): blob.extend(np.ascontiguousarray(np.asarray(a, np.float32)).ravel().tobytes())
def lin(w): put(P(w.weight).T); put(P(w.bias))
def ln(w): put(P(w.weight)); put(P(w.bias))

# patch_embed + precomputed pos_embed
put(P(tr.patch_embed.proj.weight)); put(P(tr.patch_embed.proj.bias))       # [96,3,7,7]
pos = tr._get_pos_embed((256, 256))                                        # [1,256,256,96]
put(P(pos))
# blocks
for b in tr.blocks:
    ln(b.norm1); lin(b.attn.qkv); lin(b.attn.proj)
    if b.dim != b.dim_out: lin(b.proj)                                     # shortcut projection
    ln(b.norm2); lin(b.mlp.layers[0]); lin(b.mlp.layers[1])
# neck convs (1x1)
for c in enc.neck.convs: put(P(c.conv.weight)); put(P(c.conv.bias))
open(os.path.join(HERE, "hiera_weights.bin"), "wb").write(blob)
np.frombuffer(bytes(blob), np.float32).astype(np.float16).tofile(os.path.join(HERE, "hiera_weights_fp16.bin"))

# parity refs
x = torch.sin(torch.arange(1*3*1024*1024, dtype=torch.float32).reshape(1,3,1024,1024)*0.0005)
with torch.no_grad():
    # trunk stage outputs (channels-first [B,C,H,W])
    stage_outs = tr(x)
    out = enc(x)
P(x).tofile(os.path.join(HERE, "hiera_in.bin"))
for i, s in enumerate(stage_outs): P(s).tofile(os.path.join(HERE, f"hiera_stage{i}.bin"))
fpn = out["backbone_fpn"]
for i, f in enumerate(fpn): P(f).tofile(os.path.join(HERE, f"hiera_fpn{i}.bin"))
P(out["vision_features"]).tofile(os.path.join(HERE, "hiera_vision.bin"))
print(f"hiera_weights.bin {len(blob)/1e6:.1f} MB")
print("stage outs:", [tuple(s.shape) for s in stage_outs])
print("fpn:", [tuple(f.shape) for f in fpn], "vision", tuple(out["vision_features"].shape))
