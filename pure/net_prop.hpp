// Pure-C++ SAM2 (MedSAM2) propagation wiring: condition frame t's features on the memory bank.
// Assembles memory = [spatial memories (64²/64 + maskmem_pos + temporal tpos) ; object-pointer tokens
// (each 256-d ptr split into 4×64 + temporal sine pos)] and runs memory_attention (obj ptrs excluded
// from RoPE). Verified against SAM2's _prepare_memory_conditioned_features. B=1, single object.
#pragma once
#include "net_memattn.hpp"
#include "net_hiera.hpp"     // sp2tok/tok2sp
#include <vector>
#include <cmath>

struct PropW {                                                    // constants from export_prop.py
  Tensor no_mem_embed, no_mem_pos_enc, maskmem_tpos_enc, no_obj_embed_spatial, no_obj_ptr;
  Tensor optp_w, optp_b;                                          // obj_ptr_tpos_proj (256->64)
  Tensor maskmem_pos;                                             // precomputed sine [1,64,64,64]
};
inline PropW load_prop(const std::string& dir) {
  std::string D = dir; if (!D.empty() && D.back() != '/') D += '/';
  std::ifstream f(D + "prop_weights.bin", std::ios::binary); if (!f) { printf("missing %sprop_weights.bin\n", D.c_str()); std::exit(1); }
  std::vector<float> buf; f.seekg(0, std::ios::end); size_t n = f.tellg() / 4; f.seekg(0); buf.resize(n); f.read((char*)buf.data(), n * 4);
  size_t off = 0; auto take = [&](std::vector<int64_t> sh) { int64_t m = 1; for (auto d : sh) m *= d;
    Tensor t = from_data(sh, std::vector<float>(buf.begin() + off, buf.begin() + off + m), false); off += m; return t; };
  PropW p;
  p.no_mem_embed = take({1,1,256}); p.no_mem_pos_enc = take({1,1,256}); p.maskmem_tpos_enc = take({7,1,1,64});
  p.no_obj_embed_spatial = take({1,64}); p.no_obj_ptr = take({1,256});
  for (int i = 0; i < 3; ++i) { take({256,256}); take({256}); }   // obj_ptr_proj (unused here — obj_ptr given)
  p.optp_w = take({256,64}); p.optp_b = take({64});
  p.maskmem_pos = take({1,64,64,64});
  return p;
}

// 1D sine pos emb (get_1d_sine_pe): dim=256 -> [256]. pe_dim=128; dim_t[k]=temp^(2*(k//2)/pe_dim).
inline Tensor sine1d(float pos, int64_t dim = 256, float temp = 10000.f) {
  int64_t pd = dim / 2; std::vector<float> o(dim);
  for (int64_t k = 0; k < pd; ++k) { float dt = std::pow(temp, (float)(2 * (k / 2)) / pd); float v = pos / dt;
    o[k] = std::sin(v); o[pd + k] = std::cos(v); }
  return from_data({1, dim}, o, false);
}

// condition frame t (t_pos=0 cond frame) on frame0's memory. feat1[4096,256], pos1[4096,256],
// mm0[1,64,64,64] (spatial memory), obj_ptr0[1,256]. -> conditioned pix_feat [1,256,64,64].
inline Tensor prop_condition(const Tensor& feat1, const Tensor& pos1, const Tensor& mm0, const Tensor& obj_ptr0,
                             MemW& mw, PropW& p, int t_pos = 0, float dt = 1.f, int num_frames = 2, int max_obj_ptrs = 16, int excl = 4) {
  int t_diff_max = std::min(num_frames, max_obj_ptrs) - 1;        // NOT max_obj_ptrs-1: clamped by num_frames
  // spatial memory: [4096,64] + (maskmem_pos flat + maskmem_tpos_enc[num_maskmem-t_pos-1])
  Tensor sp = sp2tok(mm0);                                        // [4096,64]
  Tensor sp_pos = add_rowvec(sp2tok(p.maskmem_pos), slice_rows(reshape(p.maskmem_tpos_enc, {7, 64}), 7 - t_pos - 1, 7 - t_pos));
  // object pointer: split 256 -> 4 tokens of 64; temporal = obj_ptr_tpos_proj(sine1d(dt/(max-1)))
  Tensor obj = reshape(obj_ptr0, {4, 64});                        // 4 tokens
  Tensor opos1 = add_rowvec(matmul(sine1d(dt / t_diff_max), p.optp_w), p.optp_b);   // [1,64]
  Tensor obj_pos = vcat({opos1, opos1, opos1, opos1});            // repeat_interleave 4 -> [4,64]
  Tensor memory = vcat({sp, obj});                               // [4100,64]
  Tensor memory_pos = vcat({sp_pos, obj_pos});
  Tensor out = memory_attention(feat1, pos1, memory, memory_pos, mw, 64, 64, 4, excl);
  return reshape(transpose2d(out), {1, 256, 64, 64});
}
