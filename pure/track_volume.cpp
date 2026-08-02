// MedSAM2 3D-volume / video track CLI (pure C++, no Python). Click one slice -> propagate the mask to
// every slice (forward + backward from the prompt slice) via the SAM2 memory bank. Outputs a mask
// overlay PNG per slice. The Hiera encoder runs once per slice (~min/slice); decoding + memory is fast.
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\track_volume.cpp
//   run:   track_volume <slice_dir> <prompt_slice> <x> <y> [ref_dir] [out_dir]
//          slice_dir = folder of PNG/JPG slices (sorted by name); x,y = click on <prompt_slice> in px.
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "net_hiera.hpp"
#include "net_sam2dec.hpp"
#include "net_memattn.hpp"
#include "net_memenc.hpp"
#include "net_prop.hpp"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
namespace fs = std::filesystem;

static Tensor preprocess(const std::string& path, int& W0, int& H0) {
  int ch; unsigned char* im = stbi_load(path.c_str(), &W0, &H0, &ch, 3);
  if (!im) { printf("cannot load %s\n", path.c_str()); std::exit(1); }
  const int S = 1024; const float mean[3] = {0.485f,0.456f,0.406f}, sd[3] = {0.229f,0.224f,0.225f};
  std::vector<float> x(3 * S * S);
  auto smp = [&](int yy,int xx,int c){ yy=std::clamp(yy,0,H0-1); xx=std::clamp(xx,0,W0-1); return (float)im[(yy*W0+xx)*3+c]; };
  for (int y=0;y<S;++y) for (int xx=0;xx<S;++xx){ float sy=(y+0.5f)*H0/S-0.5f, sx=(xx+0.5f)*W0/S-0.5f;
    int y0=(int)std::floor(sy),x0=(int)std::floor(sx); float fy=sy-y0,fx=sx-x0;
    for (int c=0;c<3;++c){ float v=smp(y0,x0,c)*(1-fx)*(1-fy)+smp(y0,x0+1,c)*fx*(1-fy)+smp(y0+1,x0,c)*(1-fx)*fy+smp(y0+1,x0+1,c)*fx*fy;
      x[(c*S+y)*S+xx]=(v/255.f-mean[c])/sd[c]; } }
  stbi_image_free(im); return from_data({1,3,S,S}, x, false);
}
static Tensor up1024(const Tensor& m){ int64_t S=256,T=1024; std::vector<float> o(T*T);
  for(int64_t y=0;y<T;++y)for(int64_t x=0;x<T;++x){ float sy=(y+0.5f)*S/T-0.5f, sx=(x+0.5f)*S/T-0.5f;
    int y0=(int)std::floor(sy),x0=(int)std::floor(sx); float fy=sy-y0,fx=sx-x0;
    auto g=[&](int yy,int xx){yy=std::max<int64_t>(0,std::min<int64_t>(S-1,yy));xx=std::max<int64_t>(0,std::min<int64_t>(S-1,xx));return m->data[yy*S+xx];};
    o[y*T+x]=g(y0,x0)*(1-fx)*(1-fy)+g(y0,x0+1)*fx*(1-fy)+g(y0+1,x0)*(1-fx)*fy+g(y0+1,x0+1)*fx*fy; }
  return from_data({1,1,T,T},o,false); }

struct FrameMem { Tensor maskmem, obj_ptr; };
static const int NUM_MASKMEM = 7, MAX_OBJ_PTRS = 16;

int main(int argc, char** argv){
  setvbuf(stdout,nullptr,_IONBF,0);
  if (argc < 5) { printf("usage: track_volume <slice_dir> <prompt_slice> <x> <y> [ref_dir] [out_dir]\n"); return 1; }
  std::string dir = argv[1]; int ps = atoi(argv[2]); float px = (float)atof(argv[3]), py = (float)atof(argv[4]);
  std::string RF = argc>5?argv[5]:"pure/ref/", OUT = argc>6?argv[6]:"track_out"; if(RF.back()!='/')RF+='/';
  fs::create_directories(OUT);
  std::vector<std::string> slices; for (auto& e : fs::directory_iterator(dir)) { auto s=e.path().string();
    auto lc=s; std::transform(lc.begin(),lc.end(),lc.begin(),::tolower);
    if (lc.size()>4 && (lc.substr(lc.size()-4)==".png"||lc.substr(lc.size()-4)==".jpg"||lc.substr(lc.size()-4)=="jpeg")) slices.push_back(s); }
  std::sort(slices.begin(), slices.end()); int N = (int)slices.size();
  if (N==0){ printf("no slices in %s\n", dir.c_str()); return 1; }
  printf("volume: %d slices, prompt slice %d @ (%.0f,%.0f)\n", N, ps, px, py);

  HieraW ew=load_hiera(RF); MemW mw=load_memattn(RF); MencW cw=load_memenc(RF); PropW pp=load_prop(RF);
  SamW dw; { std::ifstream f(RF+"dec_config.txt"); std::string k; int v; while(f>>k>>v){ if(k=="embed_dim")dw.cfg.embed=v; else if(k=="image_embed")dw.cfg.img=v; else if(k=="input_image")dw.cfg.input=v; else if(k=="num_mask_tokens")dw.cfg.mask_tokens=v; else if(k=="heads")dw.cfg.heads=v; else if(k=="tf_depth")dw.cfg.depth=v; } }
  { std::ifstream f(RF+"dec_weights.bin",std::ios::binary); f.seekg(0,std::ios::end); size_t n=f.tellg()/4; f.seekg(0); dw.buf.resize(n); f.read((char*)dw.buf.data(),n*4); }

  std::vector<FrameMem> fmem(N);              // per-slice memory (spatial + obj_ptr)
  std::vector<int> order;                     // propagation order (prompt first, then out both ways)
  order.push_back(ps);
  for (int d=1; d<N; ++d){ if (ps+d<N) order.push_back(ps+d); if (ps-d>=0) order.push_back(ps-d); }

  auto encode_mem = [&](std::vector<Tensor>& fpn, const Tensor& mask256, bool from_pts, float obj_score){
    Tensor hires = up1024(mask256);
    std::vector<float> mfm(1024*1024); for(size_t i=0;i<mfm.size();++i) mfm[i]=(1.f/(1.f+std::exp(-hires->data[i])))*20.f-10.f;
    Tensor mm = memenc_forward(fpn[2], from_data({1,1,1024,1024},mfm,false), cw, true);
    return mm;
  };
  auto save_overlay = [&](int idx, const Tensor& mask256){
    int W0,H0,ch; unsigned char* im=stbi_load(slices[idx].c_str(),&W0,&H0,&ch,3);
    std::vector<unsigned char> o(W0*H0*3);
    for(int y=0;y<H0;++y)for(int x=0;x<W0;++x){ int mx=std::clamp((int)std::round((x+0.5f)/W0*256-0.5f),0,255), my=std::clamp((int)std::round((y+0.5f)/H0*256-0.5f),0,255);
      bool fg=mask256->data[my*256+mx]>0.f; unsigned char* q=&o[(y*W0+x)*3];
      for(int c=0;c<3;++c){ float v=im[(y*W0+x)*3+c]; q[c]=(unsigned char)std::clamp(fg? v*0.5f+(c==0?255.f:60.f)*0.5f : v,0.f,255.f); } }
    char fn[512]; snprintf(fn,sizeof(fn),"%s/slice_%03d.png",OUT.c_str(),idx);
    stbi_write_png(fn,W0,H0,3,o.data(),W0*3); stbi_image_free(im);
  };

  for (int step=0; step<N; ++step){ int t=order[step]; int W0,H0;
    printf("[%d/%d] slice %d: encoding...\n", step+1, N, t);
    ew.off=0; Tensor img = preprocess(slices[t], W0, H0);
    auto fpn = hiera_forward(img, ew);
    Tensor mask_best; float obj_score;
    if (t==ps){                                                   // prompt frame: +no_mem_embed, decode(point)
      Tensor emb = reshape(transpose2d(add_rowvec(sp2tok(fpn[2]), reshape(pp.no_mem_embed,{1,256}))), {1,256,64,64});
      dw.off=0; Sam2Out o = sam2_decode(emb, fpn[0], fpn[1], {px/W0*1024.f, py/H0*1024.f}, {1}, dw);
      int best=1; for(int i=2;i<=3;++i) if(o.iou->data[i]>o.iou->data[best]) best=i;
      fmem[t].obj_ptr = obj_ptr_proj(slice_rows(o.mask_tokens,best,best+1), pp);
      mask_best = from_data({1,1,256,256}, std::vector<float>(o.masks->data.begin()+best*256*256, o.masks->data.begin()+(best+1)*256*256), false);
      obj_score = o.obj->data[0];
    } else {                                                      // propagate: condition on memory bank
      std::vector<std::pair<Tensor,int>> mems; std::vector<std::pair<Tensor,float>> ptrs;
      mems.push_back({fmem[ps].maskmem, 0});                      // cond frame (prompt) t_pos=0
      ptrs.push_back({fmem[ps].obj_ptr, (float)(t - ps)});        // signed Δt
      int tp=NUM_MASKMEM-1;                                       // recent frames get t_pos 6,5,... (nearest=6)
      for (int d=1; d<NUM_MASKMEM && tp>=1; ++d){ int pf = t - ((t>ps)?d:-d); // previous frame in propagation dir
        if (pf!=ps && pf>=0 && pf<N && fmem[pf].maskmem){ mems.push_back({fmem[pf].maskmem, tp--});
          if ((int)ptrs.size()<MAX_OBJ_PTRS) ptrs.push_back({fmem[pf].obj_ptr, (float)(t-pf)}); } }
      Tensor pix = prop_condition_multi(sp2tok(fpn[2]), sp2tok(pp.vision_pos), mems, ptrs, mw, pp, N);
      dw.off=0; Sam2Out o = sam2_decode(pix, fpn[0], fpn[1], {0.f,0.f}, {-1}, dw);   // no prompt
      fmem[t].obj_ptr = obj_ptr_proj(slice_rows(o.mask_tokens,0,1), pp);
      mask_best = from_data({1,1,256,256}, std::vector<float>(o.masks->data.begin(), o.masks->data.begin()+256*256), false);
      obj_score = o.obj->data[0];
    }
    fmem[t].maskmem = encode_mem(fpn, mask_best, t==ps, obj_score);
    save_overlay(t, mask_best);
    long area=0; for(int i=0;i<256*256;++i) if(mask_best->data[i]>0) ++area;
    printf("        obj_score=%.2f  mask %.1f%%\n", obj_score, 100.0*area/(256*256));
  }
  printf("done -> %s/slice_XXX.png\n", OUT.c_str());
  return 0;
}
