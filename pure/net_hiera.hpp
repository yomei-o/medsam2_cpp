// Pure-C++ MedSAM2 (SAM2 Hiera-T) image encoder — inference. Input [1,3,1024,1024] -> FPN features.
// Hiera hierarchical ViT: patch_embed(conv7/s4) + precomputed pos_embed + 12 MultiScaleBlocks
// (windowed; q_pool downsample at blocks 1,3,10; global at 5,7,9) -> stage outputs -> FPN neck.
// Channels-last [B,H,W,C] as [H*W,C] tokens (B=1). Weights in export_hiera.py order.
#pragma once
#include "autograd.hpp"
#include "depth_ops.hpp"     // layernorm, gelu
#include "ops2d.hpp"         // reshape, mul_scalar, softmax_rows
#include "linalg.hpp"        // matmul, transpose2d
#include "face_ops.hpp"      // relu, add_rowvec
#include "net_tinyvit.hpp"   // tv_detach, to_windows/from_windows, tok2sp/sp2tok, TvitW-style loader helpers
#include <fstream>
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>

struct HieraW {
  std::vector<float> buf; size_t off = 0;
  Tensor take(std::vector<int64_t> shape) { int64_t n = 1; for (auto d : shape) n *= d;
    Tensor t = from_data(shape, std::vector<float>(buf.begin() + off, buf.begin() + off + n), false); off += n; return t; }
};
inline HieraW load_hiera(const std::string& dir, bool fp16 = false) {
  std::string D = dir; if (!D.empty() && D.back() != '/') D += '/'; HieraW w;
  std::string fn = D + (fp16 ? "hiera_weights_fp16.bin" : "hiera_weights.bin");
  std::ifstream f(fn, std::ios::binary); if (!f) { printf("missing %s\n", fn.c_str()); std::exit(1); }
  if (fp16) { f.seekg(0, std::ios::end); size_t n = f.tellg() / 2; f.seekg(0); std::vector<uint16_t> h(n); f.read((char*)h.data(), n * 2);
    w.buf.resize(n); for (size_t i = 0; i < n; ++i) w.buf[i] = tvit_h2f(h[i]); return w; }
  f.seekg(0, std::ios::end); size_t n = f.tellg() / 4; f.seekg(0); w.buf.resize(n); f.read((char*)w.buf.data(), n * 4);
  return w;
}

// maxpool 2x2 stride2 on a [H,W,C] tile (rows=H*W) -> [(H/2)*(W/2), C]
inline Tensor maxpool2x2_hwc(const Tensor& x, int64_t H, int64_t W, int64_t C) {
  int64_t H2 = H / 2, W2 = W / 2; std::vector<float> o((size_t)H2 * W2 * C);
  for (int64_t y = 0; y < H2; ++y) for (int64_t xx = 0; xx < W2; ++xx) for (int64_t c = 0; c < C; ++c) {
    float m = -1e30f;
    for (int64_t dy = 0; dy < 2; ++dy) for (int64_t dx = 0; dx < 2; ++dx)
      m = std::max(m, x->data[(((y * 2 + dy) * W) + (xx * 2 + dx)) * C + c]);
    o[(y * W2 + xx) * C + c] = m;
  }
  return from_data({H2 * W2, C}, o, false);
}

// MultiScaleAttention on one window/tile [N=H*W, C_in] -> [N' , dim_out]. q_pool halves H,W (q only).
inline Tensor msa(const Tensor& x, int64_t H, int64_t W, const Tensor& qkvW, const Tensor& qkvB,
                  const Tensor& projW, const Tensor& projB, int heads, int64_t dim_out, bool q_pool) {
  int64_t Ch = dim_out / heads;
  Tensor qkv = add_rowvec(matmul(x, qkvW), qkvB);                  // [N, 3*dim_out]; layout [3,heads,Ch] per row
  // split q,k,v: column c index = (three*heads + head)*Ch + d  => q=[0,dim_out), k=[dim_out,2dim_out), v=...
  Tensor q = slice_cols(qkv, 0, dim_out), k = slice_cols(qkv, dim_out, 2 * dim_out), v = slice_cols(qkv, 2 * dim_out, 3 * dim_out);
  int64_t Hq = H, Wq = W;
  if (q_pool) { q = maxpool2x2_hwc(q, H, W, dim_out); Hq = H / 2; Wq = W / 2; }
  float scale = 1.f / std::sqrt((float)Ch);
  std::vector<Tensor> outs;
  for (int h = 0; h < heads; ++h) {
    Tensor qh = slice_cols(q, h * Ch, h * Ch + Ch), kh = slice_cols(k, h * Ch, h * Ch + Ch), vh = slice_cols(v, h * Ch, h * Ch + Ch);
    Tensor att = softmax_rows(mul_scalar(matmul(qh, transpose2d(kh)), scale));   // [Nq, Nk]
    outs.push_back(matmul(att, vh));                              // [Nq, Ch]
  }
  return add_rowvec(matmul(hcat(outs), projW), projB);           // [Nq, dim_out]
}

// one MultiScaleBlock. tokens x[H*W,C], returns tokens at (possibly halved) resolution.
struct BlkOut { Tensor x; int64_t H, W, C; };
inline BlkOut hiera_block(const Tensor& x, int64_t H, int64_t W, int64_t C, HieraW& w,
                          int heads, int64_t dim_out, int win, bool q_pool) {
  Tensor n1w = w.take({C}), n1b = w.take({C});
  Tensor qkvW = w.take({C, 3 * dim_out}), qkvB = w.take({3 * dim_out});
  Tensor pjW = w.take({dim_out, dim_out}), pjB = w.take({dim_out});
  Tensor spW, spB; if (C != dim_out) { spW = w.take({C, dim_out}); spB = w.take({dim_out}); }
  Tensor n2w = w.take({dim_out}), n2b = w.take({dim_out});
  Tensor f1w = w.take({dim_out, 4 * dim_out}), f1b = w.take({4 * dim_out}), f2w = w.take({4 * dim_out, dim_out}), f2b = w.take({dim_out});

  Tensor xn = layernorm(x, n1w, n1b, 1e-6f);
  // shortcut: project (if dim change) + pool (if q_pool)
  Tensor shortcut = x; int64_t sH = H, sW = W;
  if (C != dim_out) { Tensor s = add_rowvec(matmul(xn, spW), spB);
    if (q_pool) { s = maxpool2x2_hwc(s, H, W, dim_out); sH = H / 2; sW = W / 2; }
    shortcut = s; }
  // attention (windowed or global)
  Tensor a; int64_t oH, oW;
  if (win > 0) {
    Windows wd = to_windows(xn, H, W, C, win);
    std::vector<Tensor> outs;
    for (auto& t : wd.win) outs.push_back(msa(t, win, win, qkvW, qkvB, pjW, pjB, heads, dim_out, q_pool));
    int owin = q_pool ? win / 2 : win;
    oH = sH; oW = sW;                                            // output res = shortcut res
    a = from_windows(outs, oH, oW, dim_out, owin, wd.nH, wd.nW);
  } else {
    a = msa(xn, H, W, qkvW, qkvB, pjW, pjB, heads, dim_out, q_pool);
    oH = q_pool ? H / 2 : H; oW = q_pool ? W / 2 : W;
  }
  Tensor h = add(shortcut, a);                                    // residual
  Tensor m = add_rowvec(matmul(gelu(add_rowvec(matmul(layernorm(h, n2w, n2b, 1e-6f), f1w), f1b)), f2w), f2b);
  return {tv_detach(add(h, m)), oH, oW, dim_out};
}

// full Hiera encoder -> FPN features (3x [1,256,H,W]) + vision_features (64^2)
inline std::vector<Tensor> g_hiera_stages;                       // stage outputs [1,C,H,W] (debug/neck)
inline std::vector<Tensor> hiera_forward(const Tensor& img, HieraW& w) {
  // patch_embed conv7/s4/p3 -> [1,96,256,256] -> tokens [256*256,96] + pos_embed const
  Tensor pw = w.take({96, 3, 7, 7}), pb = w.take({96});
  Tensor pe = conv2d(img, pw, pb, 4, 3, 1);                      // [1,96,256,256]
  Tensor pos = w.take({256 * 256, 96});                          // precomputed [256,256,96] flattened
  Tensor x = add(sp2tok(pe), pos);                               // [65536,96]
  int64_t H = 256, W = 256, C = 96;
  const int   wins[12]  = {8,8,4,4,14,0,14,0,14,0,14,7};
  const int   heads[12] = {1,2,2,4,4,4,4,4,4,4,8,8};
  const int64_t dout[12]= {96,192,192,384,384,384,384,384,384,384,768,768};
  const bool  pool[12]  = {0,1,0,1,0,0,0,0,0,0,1,0};
  const int   stage_end[4] = {0,2,9,11};
  g_hiera_stages.clear();
  for (int i = 0; i < 12; ++i) {
    BlkOut o = hiera_block(x, H, W, C, w, heads[i], dout[i], wins[i], pool[i]);
    x = o.x; H = o.H; W = o.W; C = o.C;
    for (int s = 0; s < 4; ++s) if (stage_end[s] == i) g_hiera_stages.push_back(reshape(transpose2d(x), {1, C, H, W}));
  }
  // FPN neck: 1x1 conv each stage -> 256; top-down add on the 2 finest levels (nearest x2)
  Tensor c0w = w.take({256, 768, 1, 1}), c0b = w.take({256});    // stage3 32^2
  Tensor c1w = w.take({256, 384, 1, 1}), c1b = w.take({256});    // stage2 64^2
  Tensor c2w = w.take({256, 192, 1, 1}), c2b = w.take({256});    // stage1 128^2
  Tensor c3w = w.take({256, 96, 1, 1}), c3b = w.take({256});     // stage0 256^2
  // SAM2 FpnNeck: xs=[stage0..3]; conv[3-i] on xs[i]; top_down_levels=[2,3] => ONLY the 32^2 level
  // propagates into the 64^2 level; the 128^2 and 256^2 levels are lateral-only. interp=nearest x2.
  Tensor l32  = conv2d(g_hiera_stages[3], c0w, c0b, 1, 0, 1);    // conv0(stage3) 32^2 (out[3], dropped by scalp)
  Tensor l64  = conv2d(g_hiera_stages[2], c1w, c1b, 1, 0, 1);    // conv1(stage2) 64^2
  Tensor l128 = conv2d(g_hiera_stages[1], c2w, c2b, 1, 0, 1);    // conv2(stage1) 128^2
  Tensor l256 = conv2d(g_hiera_stages[0], c3w, c3b, 1, 0, 1);    // conv3(stage0) 256^2
  auto up2 = [](const Tensor& t) { int64_t C = t->shape[1], H = t->shape[2], W = t->shape[3];
    std::vector<float> o((size_t)C*2*H*2*W);
    for (int64_t c=0;c<C;++c) for (int64_t y=0;y<2*H;++y) for (int64_t x=0;x<2*W;++x)
      o[(c*2*H+y)*2*W+x] = t->data[(c*H+y/2)*W+x/2];
    return from_data({1,C,2*H,2*W}, o, false); };                // nearest-neighbor upsample
  Tensor o64 = add(l64, up2(l32));                               // out[2] = fpn2 (vision_features) += up(32^2)
  return {l256, l128, o64};                                      // backbone_fpn [256^2, 128^2, 64^2]
}
