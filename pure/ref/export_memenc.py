# Extract MedSAM2 memory_encoder weights + synthetic parity ref. pix_feat[1,256,64,64] + masks[1,1,1024,
# 1024] -> vision_features[1,64,64,64]. mask_downsampler(4x conv-LN2d-GELU stride2 + 1x1) + pix_feat_proj
# + fuser(2x CXBlock) + out_proj(256->64). masks sigmoided first.
import os, numpy as np, torch
from sam2.build_sam import build_sam2
HERE=os.path.dirname(__file__)
m=build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", os.path.join(HERE,"MedSAM2_latest.pt"), device="cpu"); m.eval()
me=m.memory_encoder
def P(t): return t.detach().cpu().numpy()
blob=bytearray()
def put(a): blob.extend(np.ascontiguousarray(np.asarray(a,np.float32)).ravel().tobytes())
def conv(x): put(P(x.weight)); put(P(x.bias))
def ln(x): put(P(x.weight)); put(P(x.bias))
def lin(x): put(P(x.weight).T); put(P(x.bias))
enc=me.mask_downsampler.encoder
i=0
while i < len(enc):
    if type(enc[i]).__name__=="Conv2d": conv(enc[i]); i+=1
    elif type(enc[i]).__name__=="LayerNorm2d": ln(enc[i]); i+=1
    else: i+=1
conv(me.pix_feat_proj)
for cx in me.fuser.layers:
    conv(cx.dwconv); ln(cx.norm); lin(cx.pwconv1); lin(cx.pwconv2); put(P(cx.gamma))
conv(me.out_proj)
open(os.path.join(HERE,"menc_weights.bin"),"wb").write(blob)

pix=torch.sin(torch.arange(1*256*64*64,dtype=torch.float32).reshape(1,256,64,64)*0.001)
masks=torch.cos(torch.arange(1*1*1024*1024,dtype=torch.float32).reshape(1,1,1024,1024)*0.002)
with torch.no_grad():
    out=me(pix_feat=pix, masks=masks)
P(pix).tofile(os.path.join(HERE,"menc_pix.bin")); P(masks).tofile(os.path.join(HERE,"menc_masks.bin"))
P(out["vision_features"]).tofile(os.path.join(HERE,"menc_out.bin"))
print(f"menc_weights.bin {len(blob)/1e6:.2f} MB  out {tuple(out['vision_features'].shape)}")
