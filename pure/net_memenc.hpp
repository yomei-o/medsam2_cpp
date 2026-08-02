// Pure-C++ SAM2 (MedSAM2) memory_encoder: encode a predicted mask + image features into 64-dim memory.
// mask_downsampler (4× conv-LN2d-GELU stride2 + 1×1 -> 64²/256) + pix_feat_proj + fuser (2× ConvNeXt
// CXBlock) + out_proj(256->64). masks sigmoided first. Weights in export_memenc.py order.
#pragma once
#include "net_hiera.hpp"     // conv2d, sp2tok/tok2sp, gelu, tv_detach, HieraW-style loader helpers
#include "sam_ops.hpp"       // layernorm2d
#include "depth_ops.hpp"     // scale_cols
#include <vector>
#include <cmath>

struct MencW {
  std::vector<float> buf; size_t off = 0;
  Tensor take(std::vector<int64_t> shape) { int64_t n = 1; for (auto d : shape) n *= d;
    Tensor t = from_data(shape, std::vector<float>(buf.begin() + off, buf.begin() + off + n), false); off += n; return t; }
};
inline MencW load_memenc(const std::string& dir) {
  std::string D = dir; if (!D.empty() && D.back() != '/') D += '/'; MencW w;
  std::ifstream f(D + "menc_weights.bin", std::ios::binary); if (!f) { printf("missing %smenc_weights.bin\n", D.c_str()); std::exit(1); }
  f.seekg(0, std::ios::end); size_t n = f.tellg() / 4; f.seekg(0); w.buf.resize(n); f.read((char*)w.buf.data(), n * 4);
  return w;
}

inline Tensor sigmoid_t(const Tensor& x) { std::vector<float> o(x->data.size());
  for (size_t i = 0; i < o.size(); ++i) o[i] = 1.f / (1.f + std::exp(-x->data[i])); return from_data(x->shape, o, false); }

// ConvNeXt CXBlock on [1,C,H,W]: dwconv7x7 -> LN2d -> pwconv1(C->4C) -> GELU -> pwconv2 -> gamma -> +x
inline Tensor cxblock(const Tensor& x, MencW& w, int64_t C, int64_t H, int64_t Wd) {
  Tensor dww = w.take({C,1,7,7}), dwb = w.take({C}), nw = w.take({C}), nb = w.take({C});
  Tensor p1w = w.take({C,4*C}), p1b = w.take({4*C}), p2w = w.take({4*C,C}), p2b = w.take({C}), gamma = w.take({C});
  Tensor h = layernorm2d(conv2d(x, dww, dwb, 1, 3, C), nw, nb, 1e-6f);   // dwconv + LN2d
  Tensor t = sp2tok(h);                                                  // [HW,C]
  t = add_rowvec(matmul(gelu(add_rowvec(matmul(t, p1w), p1b)), p2w), p2b);
  t = scale_cols(t, gamma);                                             // layer scale
  return add(x, tok2sp(t, C, H, Wd));
}

// memory_encoder: pix_feat[1,256,64,64] + masks[1,1,1024,1024] -> memory features [1,64,64,64]
inline Tensor memenc_forward(const Tensor& pix_feat, const Tensor& masks_logits, MencW& w, bool skip_sigmoid = false) {
  Tensor mk = skip_sigmoid ? masks_logits : sigmoid_t(masks_logits);   // track: mask already sigmoid*20-10
  // mask_downsampler: 4x (conv3/s2/p1 -> LN2d -> GELU), then 1x1
  int64_t chin = 1; const int64_t chs[4] = {4,16,64,256};
  Tensor m = mk;
  for (int i = 0; i < 4; ++i) { Tensor cw = w.take({chs[i], chin, 3, 3}), cb = w.take({chs[i]}), lw = w.take({chs[i]}), lb = w.take({chs[i]});
    m = gelu(layernorm2d(conv2d(m, cw, cb, 2, 1, 1), lw, lb, 1e-6f)); chin = chs[i]; }
  Tensor c4w = w.take({256,256,1,1}), c4b = w.take({256});
  m = conv2d(m, c4w, c4b, 1, 0, 1);                                     // [1,256,64,64]
  // pix_feat_proj + fuse
  Tensor pw = w.take({256,256,1,1}), pb = w.take({256});
  Tensor x = add(conv2d(pix_feat, pw, pb, 1, 0, 1), m);
  for (int i = 0; i < 2; ++i) x = cxblock(x, w, 256, 64, 64);          // fuser
  Tensor ow = w.take({64,256,1,1}), ob = w.take({64});
  return conv2d(x, ow, ob, 1, 0, 1);                                   // [1,64,64,64]
}
