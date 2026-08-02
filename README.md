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

## Plan (staged parity, like the SAM port)
1. **Hiera image encoder** → parity vs PyTorch (the big new piece)
2. prompt encoder + mask decoder, **single-image** path (no memory) → single-frame inference
3. **memory attention + memory encoder** → 3D volume / video (the temporal core)
4. training ; 5. WASM ; 6. GPU (cuBLAS seam)

Reuses from medsam_cpp: engine (`autograd/backend/ops2d/linalg/...`) + `sam_ops`/`sam_loss`/`net_sam`
(decoder pieces). New: `net_hiera.hpp`, memory attention, `pure/ref/*`.

License: own code BSD-3-Clause; bundled deps keep their licenses.
