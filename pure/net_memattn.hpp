// Pure-C++ SAM2 (MedSAM2) memory_attention: 4 layers of RoPE self-attn + RoPE cross-attn(memory) + MLP.
// The temporal/3D core — the current frame's 64x64 features attend to the memory bank. B=1, num_heads=1
// (so everything is [N,256] 2D). memory is 64-dim (k/v proj 64->256). RoPE = 2D axial rotary (no weights).
// Weights in export_memattn.py order. curr [N=4096,256], memory [M,64], + pos encodings.
#pragma once
#include "net_sam.hpp"       // SamW-style helpers, layernorm, matmul, softmax_rows, add, relu, add_rowvec
#include <vector>
#include <cmath>

struct MemW {
  std::vector<float> buf; size_t off = 0;
  Tensor take(std::vector<int64_t> shape) { int64_t n = 1; for (auto d : shape) n *= d;
    Tensor t = from_data(shape, std::vector<float>(buf.begin() + off, buf.begin() + off + n), false); off += n; return t; }
};
inline MemW load_memattn(const std::string& dir) {
  std::string D = dir; if (!D.empty() && D.back() != '/') D += '/'; MemW w;
  std::ifstream f(D + "mem_weights.bin", std::ios::binary); if (!f) { printf("missing %smem_weights.bin\n", D.c_str()); std::exit(1); }
  f.seekg(0, std::ios::end); size_t n = f.tellg() / 4; f.seekg(0); w.buf.resize(n); f.read((char*)w.buf.data(), n * 4);
  return w;
}

// 2D axial RoPE on x[N, dim] over an HxW grid (token n at x=n%W, y=n/W). Rotates pairs (x[2j],x[2j+1]).
// freqs[k] = theta^(-4k/dim), k in [0,dim/4). For pair j<dim/4: angle = px*freqs[j]; else py*freqs[j-dim/4].
inline Tensor rope2d(const Tensor& x, int64_t H, int64_t W, float theta = 10000.f) {
  int64_t N = x->shape[0], dim = x->shape[1], q = dim / 4, HW = H * W;
  std::vector<float> freqs(q); for (int64_t k = 0; k < q; ++k) freqs[k] = 1.f / std::pow(theta, (float)(4 * k) / dim);
  std::vector<float> o((size_t)N * dim);
  for (int64_t n = 0; n < N; ++n) { int64_t pn = n % HW; float px = (float)(pn % W), py = (float)(pn / W); const float* xr = &x->data[n * dim]; float* orow = &o[n * dim];   // cyclic (repeat_freqs_k for multi-frame memory)
    for (int64_t j = 0; j < dim / 2; ++j) {
      float ang = (j < q) ? px * freqs[j] : py * freqs[j - q];
      float c = std::cos(ang), s = std::sin(ang), a = xr[2 * j], b = xr[2 * j + 1];
      orow[2 * j] = a * c - b * s; orow[2 * j + 1] = a * s + b * c;
    }
  }
  return from_data({N, dim}, o, false);
}

// attention (heads=1): q[Nq,256], k[Nk,256], v[Nk,256] -> [Nq,256]. scale 1/sqrt(256).
inline Tensor mem_attn(const Tensor& q, const Tensor& k, const Tensor& v) {
  float scale = 1.f / std::sqrt((float)q->shape[1]);
  return matmul(softmax_rows(mul_scalar(matmul(q, transpose2d(k)), scale)), v);
}

// full memory_attention: curr[N,256], curr_pos[N,256], memory[M,64], memory_pos[M,64] -> [N,256]
inline Tensor memory_attention(const Tensor& curr, const Tensor& curr_pos, const Tensor& memory, const Tensor& memory_pos,
                               MemW& w, int64_t H = 64, int64_t Wd = 64, int layers = 4, int64_t num_k_exclude_rope = 0) {
  Tensor out = add(curr, mul_scalar(curr_pos, 0.1f));               // pos_enc_at_input
  for (int L = 0; L < layers; ++L) {
    // self-attn: q=k=rope(norm1(tgt)); v=norm1(tgt) (pos_enc_at_attn=False)
    Tensor sqW = w.take({256, 256}), sqB = w.take({256}), skW = w.take({256, 256}), skB = w.take({256});
    Tensor svW = w.take({256, 256}), svB = w.take({256}), soW = w.take({256, 256}), soB = w.take({256});
    Tensor n1w = w.take({256}), n1b = w.take({256});
    // cross-attn (memory 64-dim): q=rope(norm2 q_proj); k=rope(k_proj(memory+mempos)); v=v_proj(memory)
    Tensor cqW = w.take({256, 256}), cqB = w.take({256}), ckW = w.take({64, 256}), ckB = w.take({256});
    Tensor cvW = w.take({64, 256}), cvB = w.take({256}), coW = w.take({256, 256}), coB = w.take({256});
    Tensor n2w = w.take({256}), n2b = w.take({256});
    Tensor l1w = w.take({256, 2048}), l1b = w.take({2048}), l2w = w.take({2048, 256}), l2b = w.take({256});
    Tensor n3w = w.take({256}), n3b = w.take({256});

    Tensor t2 = layernorm(out, n1w, n1b, 1e-5f);
    Tensor sq = rope2d(add_rowvec(matmul(t2, sqW), sqB), H, Wd);
    Tensor sk = rope2d(add_rowvec(matmul(t2, skW), skB), H, Wd);
    Tensor sv = add_rowvec(matmul(t2, svW), svB);
    out = add(out, add_rowvec(matmul(mem_attn(sq, sk, sv), soW), soB));

    Tensor c2 = layernorm(out, n2w, n2b, 1e-5f);
    Tensor cq = rope2d(add_rowvec(matmul(c2, cqW), cqB), H, Wd);
    Tensor km = add(memory, memory_pos);                            // pos_enc_at_cross_attn_keys=True
    Tensor ckp = add_rowvec(matmul(km, ckW), ckB);                  // [Nk,256]
    int64_t Nk = ckp->shape[0], Nrope = Nk - num_k_exclude_rope;    // obj-pointer tokens (last exclude) skip RoPE
    Tensor ck = num_k_exclude_rope > 0
              ? vcat({rope2d(slice_rows(ckp, 0, Nrope), H, Wd), slice_rows(ckp, Nrope, Nk)})
              : rope2d(ckp, H, Wd);
    Tensor cv = add_rowvec(matmul(memory, cvW), cvB);
    out = add(out, add_rowvec(matmul(mem_attn(cq, ck, cv), coW), coB));

    Tensor m = add_rowvec(matmul(relu(add_rowvec(matmul(layernorm(out, n3w, n3b, 1e-5f), l1w), l1b)), l2w), l2b);
    out = add(out, m);
  }
  Tensor nw = w.take({256}), nb = w.take({256});
  return layernorm(out, nw, nb, 1e-5f);
}
