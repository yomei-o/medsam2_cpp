// MedSAM2 single-image inference (pure C++, no Python): medical image + point -> Hiera encoder ->
// SAM2 mask decoder -> mask overlay PNG. Preprocess = resize to 1024² + /255 + ImageNet normalize
// (SAM2 standard). Single frame (no memory). Encoder is slow (~min); decoder is fast.
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\infer_medsam2.cpp
//   run:   infer_medsam2 <img> <x> <y> [out.png] [ref_dir]   (x,y = click in original pixels)
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "net_hiera.hpp"
#include "net_sam2dec.hpp"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <fstream>

int main(int argc, char** argv) {
  if (argc < 4) { printf("usage: infer_medsam2 <img> <x> <y> [out.png] [ref_dir]\n"); return 1; }
  std::string src = argv[1]; float px = (float)atof(argv[2]), py = (float)atof(argv[3]);
  std::string out = argc > 4 ? argv[4] : "medsam2_mask.png", RF = argc > 5 ? argv[5] : "pure/ref/";
  if (RF.back() != '/') RF += '/';
  int W0, H0, ch; unsigned char* im = stbi_load(src.c_str(), &W0, &H0, &ch, 3);
  if (!im) { printf("cannot load %s\n", src.c_str()); return 1; }

  const int S = 1024; const float mean[3] = {0.485f,0.456f,0.406f}, sd[3] = {0.229f,0.224f,0.225f};
  std::vector<float> x(3 * S * S);
  auto samp = [&](int yy,int xx,int c){ yy=std::clamp(yy,0,H0-1); xx=std::clamp(xx,0,W0-1); return (float)im[(yy*W0+xx)*3+c]; };
  for (int y=0;y<S;++y) for (int xx=0;xx<S;++xx){ float sy=(y+0.5f)*H0/S-0.5f, sx=(xx+0.5f)*W0/S-0.5f;
    int y0=(int)std::floor(sy),x0=(int)std::floor(sx); float fy=sy-y0,fx=sx-x0;
    for (int c=0;c<3;++c){ float v=samp(y0,x0,c)*(1-fx)*(1-fy)+samp(y0,x0+1,c)*fx*(1-fy)+samp(y0+1,x0,c)*(1-fx)*fy+samp(y0+1,x0+1,c)*fx*fy;
      x[(c*S+y)*S+xx]=(v/255.f-mean[c])/sd[c]; } }

  printf("MedSAM2: encoding (Hiera, ~min)...\n"); fflush(stdout);
  HieraW ew = load_hiera(RF);
  auto fpn = hiera_forward(from_data({1,3,S,S}, x), ew);            // [256²,128²,64²]
  // decoder weights (dec_weights.bin + dec_config.txt)
  SamW dw; { std::ifstream f(RF+"dec_config.txt"); std::string k; int v; while(f>>k>>v){ if(k=="embed_dim")dw.cfg.embed=v; else if(k=="image_embed")dw.cfg.img=v; else if(k=="input_image")dw.cfg.input=v; else if(k=="num_mask_tokens")dw.cfg.mask_tokens=v; else if(k=="heads")dw.cfg.heads=v; else if(k=="tf_depth")dw.cfg.depth=v; } }
  { std::ifstream f(RF+"dec_weights.bin",std::ios::binary); f.seekg(0,std::ios::end); size_t n=f.tellg()/4; f.seekg(0); dw.buf.resize(n); f.read((char*)dw.buf.data(),n*4); }

  std::vector<float> pts = {px/W0*S, py/H0*S}; std::vector<int> lab = {1};    // click -> 1024 space
  Sam2Out o = sam2_decode(fpn[2], fpn[0], fpn[1], pts, lab, dw);    // image_embed=64², hr256, hr128
  int best = 1; for (int i = 2; i <= 3; ++i) if (o.iou->data[i] > o.iou->data[best]) best = i;
  printf("iou = [%.3f %.3f %.3f] obj=%.2f -> mask %d\n", o.iou->data[1],o.iou->data[2],o.iou->data[3], o.obj->data[0], best);
  const float* m = &o.masks->data[best * 256 * 256];               // 256² logits (covers full 1024)

  std::vector<unsigned char> outimg(W0*H0*3); long area=0;
  for (int y=0;y<H0;++y) for (int xx=0;xx<W0;++xx){ int mx=std::clamp((int)std::round((xx+0.5f)/W0*256-0.5f),0,255), my=std::clamp((int)std::round((y+0.5f)/H0*256-0.5f),0,255);
    bool fg=m[my*256+mx]>0.f; if(fg)++area; unsigned char*p=&outimg[(y*W0+xx)*3];
    for(int c=0;c<3;++c){ float v=im[(y*W0+xx)*3+c]; p[c]=(unsigned char)std::clamp(fg? v*0.5f+(c==0?255.f:60.f)*0.5f : v,0.f,255.f); } }
  for (int d=-6;d<=6;++d){ int cx=(int)px,cy=(int)py; auto Pt=[&](int yy,int xx){ if(xx>=0&&xx<W0&&yy>=0&&yy<H0){unsigned char*p=&outimg[(yy*W0+xx)*3];p[0]=0;p[1]=255;p[2]=0;} }; Pt(cy,cx+d);Pt(cy+d,cx); }
  stbi_write_png(out.c_str(), W0, H0, 3, outimg.data(), W0*3);
  printf("mask area = %ld (%.1f%%) -> wrote %s\n", area, 100.0*area/(W0*H0), out.c_str());
  stbi_image_free(im); return 0;
}
