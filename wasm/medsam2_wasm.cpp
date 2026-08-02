// MedSAM2 in WASM: encode a medical image ONCE (Hiera encoder, ~tens of seconds with Eigen+SIMD), then
// segment per click. Single-frame SAM2 (no memory) — same path as pure/infer_medsam2.cpp.
//   fn_ready()                 load fp16 weights from /medsam2/ (Hiera + SAM2 decoder)
//   fn_encode(rgba,w,h)        SAM2 preprocess (resize 1024 + /255 + ImageNet norm) + Hiera -> FPN feats
//   fn_decode(px,py) -> float* point prompt (orig px,py) -> 256x256 best-IoU mask logits (>0 in JS)
//   fn_iou()                   predicted IoU of the returned mask
#include "net_hiera.hpp"
#include "net_sam2dec.hpp"
#include <emscripten/emscripten.h>
#include <vector>
#include <algorithm>
#include <cmath>

static HieraW* g_ew = nullptr; static SamW* g_dw = nullptr;
static Tensor g_f0 = nullptr, g_f1 = nullptr, g_f2 = nullptr;   // FPN [256²,128²,64²]
static int g_W0 = 0, g_H0 = 0;
static std::vector<float> g_mask(256 * 256); static float g_iou = 0.f;

static void reset_dec() { g_dw->off = 0; g_dw->ti = 0; g_dw->cache.clear(); }   // decoder reader rewind

extern "C" {

EMSCRIPTEN_KEEPALIVE int fn_ready() {
  if (!g_ew) g_ew = new HieraW(load_hiera("/medsam2/", true));
  if (!g_dw) g_dw = new SamW(load_sam_decoder("/medsam2/", true));
  return (g_ew && g_dw) ? 1 : 0;
}
EMSCRIPTEN_KEEPALIVE float fn_iou() { return g_iou; }

EMSCRIPTEN_KEEPALIVE int fn_encode(unsigned char* rgba, int w, int h) {
  if (!g_ew) fn_ready();
  g_ew->off = 0; g_W0 = w; g_H0 = h;
  const int S = 1024; const float mean[3] = {0.485f,0.456f,0.406f}, sd[3] = {0.229f,0.224f,0.225f};
  std::vector<float> x(3 * S * S);
  auto samp = [&](int yy,int xx,int c){ yy=std::clamp(yy,0,h-1); xx=std::clamp(xx,0,w-1); return (float)rgba[((size_t)yy*w+xx)*4+c]; };
  for (int y=0;y<S;++y) for (int xx=0;xx<S;++xx){ float sy=(y+0.5f)*h/S-0.5f, sx=(xx+0.5f)*w/S-0.5f;
    int y0=(int)std::floor(sy),x0=(int)std::floor(sx); float fy=sy-y0,fx=sx-x0;
    for (int c=0;c<3;++c){ float v=samp(y0,x0,c)*(1-fx)*(1-fy)+samp(y0,x0+1,c)*fx*(1-fy)+samp(y0+1,x0,c)*(1-fx)*fy+samp(y0+1,x0+1,c)*fx*fy;
      x[((size_t)c*S+y)*S+xx]=(v/255.f-mean[c])/sd[c]; } }
  auto fpn = hiera_forward(from_data({1,3,S,S}, x, false), *g_ew);
  g_f0 = fpn[0]; g_f1 = fpn[1]; g_f2 = fpn[2];
  return 1;
}

EMSCRIPTEN_KEEPALIVE float* fn_decode(float px, float py) {
  if (!g_f2) return g_mask.data();
  reset_dec();
  std::vector<float> pts = { px / g_W0 * 1024.f, py / g_H0 * 1024.f }; std::vector<int> lab = {1};
  Sam2Out o = sam2_decode(g_f2, g_f0, g_f1, pts, lab, *g_dw);
  int best = 1; for (int i = 2; i <= 3; ++i) if (o.iou->data[i] > o.iou->data[best]) best = i;
  g_iou = o.iou->data[best];
  const float* m = &o.masks->data[best * 256 * 256];
  std::copy(m, m + 256 * 256, g_mask.begin());
  return g_mask.data();
}

}  // extern "C"
