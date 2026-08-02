// Pure-C++ SAM2 (MedSAM2) prompt encoder + mask decoder, single-image path (no memory).
// = SAM's decoder + obj_score_token (prepended) + high-res feature fusion (conv_s0/conv_s1 into the two
// upscaling ConvTs) + an object-score head. Reuses net_sam.hpp helpers. B=1. Weights in
// export_medsam2_dec.py order. Inputs: image_embed 64²/256 + raw high-res fpn (256²/256, 128²/256).
#pragma once
#include "net_sam.hpp"       // SamW, AttnW, read_attn, attention, pe_encode, + ops
#include "sam_ops.hpp"       // layernorm2d
#include <vector>

struct Sam2Out { Tensor masks, iou, obj, mask_tokens; };
inline Sam2Out sam2_decode(const Tensor& img_embed, const Tensor& hr256, const Tensor& hr128,
                           const std::vector<float>& pts, const std::vector<int>& labels, SamW& w) {
  auto& c = w.cfg; int64_t E = c.embed, S = c.img, HW = S * S, Np = (int64_t)labels.size();
  // ---- prompt encoder (same as SAM) ----
  Tensor G = w.take({2, E / 2});
  Tensor ptemb[4]; for (int i = 0; i < 4; ++i) ptemb[i] = w.take({1, E});
  Tensor not_a_point = w.take({1, E}), no_mask = w.take({1, E});
  Tensor mc0w = w.take({4,1,2,2}), mc0b = w.take({4}), ml1w = w.take({4}), ml1b = w.take({4});
  Tensor mc3w = w.take({16,4,2,2}), mc3b = w.take({16}), ml4w = w.take({16}), ml4b = w.take({16});
  Tensor mc6w = w.take({E,16,1,1}), mc6b = w.take({E});
  std::vector<float> pc(Np * 2);
  for (int64_t i = 0; i < Np; ++i) { pc[i*2] = (pts[i*2]+0.5f)/c.input; pc[i*2+1] = (pts[i*2+1]+0.5f)/c.input; }
  Tensor ppe = pe_encode(pc, Np, G);
  std::vector<Tensor> srows;
  for (int64_t i = 0; i < Np; ++i) srows.push_back(labels[i] < 0 ? not_a_point            // label -1 -> not_a_point (pe zeroed)
                                                                 : add(slice_rows(ppe, i, i+1), ptemb[labels[i]]));
  srows.push_back(not_a_point);                                  // pad point (pad=True when no box)
  Tensor sparse = srows.size()==1 ? srows[0] : vcat(srows);
  std::vector<float> gc(HW*2);
  for (int64_t h=0; h<S; ++h) for (int64_t ww=0; ww<S; ++ww){ gc[(h*S+ww)*2]=(ww+0.5f)/S; gc[(h*S+ww)*2+1]=(h+0.5f)/S; }
  Tensor image_pe = pe_encode(gc, HW, G);

  // ---- mask decoder: obj_score + iou + mask tokens ----
  Tensor obj_token = w.take({1,E}), iou_token = w.take({1,E}), mask_tokens = w.take({c.mask_tokens,E});
  Tensor tokens = vcat({obj_token, iou_token, mask_tokens, sparse});   // [6+Np, 256]
  Tensor img_tok = add_rowvec(transpose2d(reshape(img_embed, {E, HW})), no_mask);
  Tensor queries = tokens, keys = img_tok, query_pe = tokens, key_pe = image_pe;
  for (int L = 0; L < c.depth; ++L) {
    AttnW self_a = read_attn(w, E, E); Tensor n1w=w.take({E}), n1b=w.take({E});
    AttnW cross_ti = read_attn(w, E, E/2); Tensor n2w=w.take({E}), n2b=w.take({E});
    int64_t MD = 2048;
    Tensor l1w=w.take({E,MD}), l1b=w.take({MD}), l2w=w.take({MD,E}), l2b=w.take({E});
    Tensor n3w=w.take({E}), n3b=w.take({E}), n4w=w.take({E}), n4b=w.take({E});
    AttnW cross_it = read_attn(w, E, E/2);
    if (L==0) queries = attention(queries, queries, queries, self_a, c.heads);
    else { Tensor q=add(queries,query_pe); queries=add(queries, attention(q,q,queries,self_a,c.heads)); }
    queries = layernorm(queries, n1w, n1b, 1e-5f);
    { Tensor q=add(queries,query_pe), k=add(keys,key_pe); queries=add(queries, attention(q,k,keys,cross_ti,c.heads)); }
    queries = layernorm(queries, n2w, n2b, 1e-5f);
    { Tensor h=add_rowvec(matmul(queries,l1w),l1b); h=add_rowvec(matmul(relu(h),l2w),l2b); queries=add(queries,h); }
    queries = layernorm(queries, n3w, n3b, 1e-5f);
    { Tensor q=add(keys,key_pe), k=add(queries,query_pe); keys=add(keys, attention(q,k,queries,cross_it,c.heads)); }
    keys = layernorm(keys, n4w, n4b, 1e-5f);
  }
  AttnW fin = read_attn(w, E, E/2); Tensor nfw=w.take({E}), nfb=w.take({E});
  { Tensor q=add(queries,query_pe), k=add(keys,key_pe); queries=add(queries, attention(q,k,keys,fin,c.heads)); }
  queries = layernorm(queries, nfw, nfb, 1e-5f);
  Tensor obj_out = slice_rows(queries, 0, 1);                          // hs[0]
  Tensor iou_out = slice_rows(queries, 1, 2);                          // hs[1]
  Tensor mask_out = slice_rows(queries, 2, 2 + c.mask_tokens);         // hs[2:6]

  // ---- high-res fusion upscaling ----
  Tensor ct0w=w.take({E,E/4,2,2}), ct0b=w.take({E/4}), ul1w=w.take({E/4}), ul1b=w.take({E/4});
  Tensor ct3w=w.take({E/4,E/8,2,2}), ct3b=w.take({E/8});
  Tensor cs0w=w.take({E/8,E,1,1}), cs0b=w.take({E/8});                 // conv_s0 256->32
  Tensor cs1w=w.take({E/4,E,1,1}), cs1b=w.take({E/4});                 // conv_s1 256->64
  Tensor feat_s0 = conv2d(hr256, cs0w, cs0b, 1, 0, 1);                 // [1,32,256,256]
  Tensor feat_s1 = conv2d(hr128, cs1w, cs1b, 1, 0, 1);                 // [1,64,128,128]
  Tensor src = reshape(transpose2d(keys), {1, E, S, S});
  Tensor u = gelu(layernorm2d(add(convtranspose2d(src, ct0w, ct0b, 2), feat_s1), ul1w, ul1b, 1e-6f));  // 128²/64
  u = gelu(add(convtranspose2d(u, ct3w, ct3b, 2), feat_s0));          // 256²/32
  int64_t US = 4 * S, C32 = E / 8;

  std::vector<Tensor> hyper;
  for (int i = 0; i < c.mask_tokens; ++i) {
    Tensor h0w=w.take({E,E}), h0b=w.take({E}), h1w=w.take({E,E}), h1b=w.take({E}), h2w=w.take({E,C32}), h2b=w.take({C32});
    Tensor mt = slice_rows(mask_out, i, i+1);
    Tensor h = relu(add_rowvec(matmul(mt,h0w),h0b)); h = relu(add_rowvec(matmul(h,h1w),h1b)); h = add_rowvec(matmul(h,h2w),h2b);
    hyper.push_back(h);
  }
  Tensor masks = reshape(matmul(vcat(hyper), reshape(u, {C32, US*US})), {c.mask_tokens, US, US});
  // iou head + obj-score head (MLP 256->256->256->{4,1})
  Tensor i0w=w.take({E,E}), i0b=w.take({E}), i1w=w.take({E,E}), i1b=w.take({E}), i2w=w.take({E,c.mask_tokens}), i2b=w.take({c.mask_tokens});
  Tensor ih = relu(add_rowvec(matmul(iou_out,i0w),i0b)); ih=relu(add_rowvec(matmul(ih,i1w),i1b));
  Tensor iou = sigmoid(add_rowvec(matmul(ih,i2w),i2b));                // SAM2 iou head: sigmoid_output=True
  Tensor o0w=w.take({E,E}), o0b=w.take({E}), o1w=w.take({E,E}), o1b=w.take({E}), o2w=w.take({E,1}), o2b=w.take({1});
  Tensor oh = relu(add_rowvec(matmul(obj_out,o0w),o0b)); oh=relu(add_rowvec(matmul(oh,o1w),o1b));
  Tensor obj = add_rowvec(matmul(oh,o2w),o2b);
  return {masks, iou, obj, mask_out};
}
