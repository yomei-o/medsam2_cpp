# medsam2_cpp — pure C++ MedSAM2 (medical Segment Anything 2) — WIP

[MedSAM2](https://github.com/bowang-lab/MedSAM2) (promptable segmentation for 3D medical volumes and
videos) ported to a dependency-free C++ autograd engine — no PyTorch at run time. Same approach as the
sibling repos (segment_anything_cpp, medsam_cpp, depth_anything_cpp, facenet/lpr).

**MedSAM2 = SAM2 fine-tuned on medical data.** Unlike MedSAM (which was just SAM ViT-B), SAM2 is a
different, larger architecture:
- **Hiera** hierarchical ViT image encoder (+ FPN neck, multi-scale features) — NOT ViT-B/TinyViT
- **memory attention + memory bank** — each slice/frame attends to neighbors for 3D/temporal consistency
- **memory encoder** — encodes the predicted mask + features into memory
- prompt encoder + mask decoder (SAM-derived, with high-res features + object score)

Reference: **Hiera-T** variant (smallest, most portable). Checkpoint from `wanglab/MedSAM2`, built on the
`sam2` package.

## Status (staged parity, like the SAM port)
1. ✅ **Hiera image encoder** (trunk + FPN neck) → parity: stages 1.8e-6..2.4e-5, FPN 2.8e-7..1.8e-6 MATCH
2. ✅ **prompt encoder + SAM2 mask decoder** (single-image, no memory) → masks 8.4e-5, iou 2.4e-7, obj exact MATCH.
   `infer_medsam2 <img> <x> <y>` = single-frame click→mask (Hiera + SAM2 decoder).
3. ✅ **memory_attention (RoPE)** + **memory_encoder** + **propagation conditioning** → parity (3.29e-5 / 1.19e-6 / 2.53e-5 MATCH)
4. ✅ **full multi-frame track** (`pure/track_medsam2.cpp`) — end-to-end 3D propagation reproduces PyTorch (obj_ptr 2e-6, mm0 1.7e-5, frame1 mask 3.2e-4 MATCH)

**MedSAM2 (SAM2) is complete in pure C++**: Hiera encoder + SAM2 decoder + memory attention (RoPE) +
memory encoder + the full frame-to-frame track loop, all matching PyTorch — i.e. click one slice, it
propagates to the rest of the volume/video. Next (optional): a real-volume track CLI, training, WASM, GPU.
4. ⏭ training ; 5. WASM ; 6. GPU (cuBLAS seam)

Reuses from medsam_cpp: engine (`autograd/backend/ops2d/linalg/...`) + `sam_ops`/`sam_loss`/`net_sam`
(decoder pieces). New: `net_hiera.hpp`, memory attention, `pure/ref/*`.

License: own code BSD-3-Clause; bundled deps keep their licenses.
