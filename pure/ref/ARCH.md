# MedSAM2 (SAM2 Hiera-T) — architecture for the C++ port

Extracted from MedSAM2_latest.pt (sam2.1_hiera_t) via inspect. Input 1024². B=1. Total ~40M:
image_encoder.trunk 26.85M (Hiera) + neck 0.37M + sam_mask_decoder 4.22M + sam_prompt_encoder 0.01M +
memory_attention 5.92M + memory_encoder 1.38M + obj_ptr/misc ~0.5M.

## Hiera-T image encoder (trunk) — STAGE 1 target
patch_embed: Conv2d(3,96,k7,s4,p3) -> [1,96,256,256] -> permute [B,H,W,C]=[1,256,256,96].
pos_embed [1,96,7,7] BICUBIC-interp to (256,256) + pos_embed_window [1,96,8,8] tiled (256/8=32×) ->
  add -> [B,256,256,96].  (Export the pre-computed [256,256,96] constant to avoid bicubic in C++.)

12 MultiScaleBlocks. Per block (window / heads / dim_out), q_pool at blocks 1,3,10 (q_stride 2):
  b0 w8  h1 d96 | b1 w8  h2 d192 POOL | b2 w4 h2 d192 | b3 w4 h4 d384 POOL |
  b4 w14 h4 d384 | b5 w0(GLOBAL) h4 d384 | b6 w14 | b7 w0 | b8 w14 | b9 w0 |
  b10 w14 h8 d768 POOL | b11 w7 h8 d768.   dims [96,192,384,768]; stage_ends [0,2,9,11].
Resolutions: 256²(96) -> [b1 pool] 128²(192) -> [b3 pool] 64²(384) -> [b10 pool] 32²(768).

MultiScaleBlock.forward(x[B,H,W,C]):
  shortcut=x; x=norm1(x)(LayerNorm eps1e-6, channels-last);
  if dim!=dim_out: shortcut = do_pool(proj(x), maxpool2x2)   # proj Linear dim->dim_out, then pool
  if window>0: x,pad = window_partition(x, window)           # [nW, ws, ws, C]
  x = attn(x)                                                 # MultiScaleAttention (pools q inside)
  if q_stride: window//=2; H,W=shortcut.shape; recompute pad_hw
  if window>0: x = window_unpartition(x, window(adjusted), pad_hw, (H,W))
  x = shortcut + x
  x = x + mlp(norm2(x))                                       # MLP 2-layer (dim_out->4*dim_out->dim_out, GELU)

MultiScaleAttention.forward(x[B,H,W,C], heads, q_pool?):
  qkv=Linear(dim,3*dim_out)(x) -> [B,HW,3,heads,Ch]; unbind q,k,v [B,HW,heads,Ch]
  if q_pool: q=maxpool2x2(q as [B,H,W,heads*Ch]) -> [B,H/2·W/2,heads,Ch]   # ONLY q pooled; k,v full
  SDPA(q,k,v) = softmax(q·kᵀ/√Ch)·v  (no rel-pos, no bias); -> [B,H'W',dim_out]; proj Linear.
  (q pooled + k,v full -> output length = q's downsampled length -> block downsamples.)
window_partition/unpartition: standard [B,H,W,C]<->[nW,ws,ws,C] with zero-pad. NOTE q_pool blocks
  unpartition with window//2 because q halved the per-window size.

## FpnNeck (0.37M)
4× Conv2d(1×1) projecting the 4 stage outputs to 256: convs.0 768->256, .1 384->256, .2 192->256,
  .3 96->256. Top-down FPN, fpn_interp="nearest", top_down_levels=[2,3] (only 2 finest levels fused).
Returns backbone_fpn (4 feats @256) + pos. Decoder uses vision_features (64²/256) + high-res (128²,256²).

## Remaining (later stages)
- sam_prompt_encoder + sam_mask_decoder (SAM-derived; decoder uses high-res feats + object-score head).
- memory_attention (5.92M): each frame/slice attends to memory bank (self+cross attn, RoPE).
- memory_encoder (1.38M): encodes predicted mask + features into memory.
- obj_ptr / no_mem_embed / temporal pos enc: the video/3D memory machinery.

## New C++ ops vs the SAM port
- maxpool2d 2×2 s2 (have maxpool in autograd) ; LayerNorm channels-last (have) ; window part/unpart
  (have from vitb) ; plain SDPA softmax attn (have) ; MLP (have). pos_embed precomputed (export).
- Later: RoPE (rotary pos emb) for memory attention (new).
