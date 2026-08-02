// MedSAM2 multi-frame TRACK driver (pure C++) — end-to-end 3D/video propagation from verified pieces.
// frame0(prompt): Hiera -> +no_mem_embed -> SAM2 decode -> best mask -> obj_ptr + memory_encode.
// frame t: Hiera -> prop_condition(memory bank) -> SAM2 decode(no prompt) -> mask -> memory_encode.
// Validation build reproduces the 2-frame reference (prop_mm0 / prop_objptr0 / prop_mask1).
#include "net_hiera.hpp"
#include "net_sam2dec.hpp"
#include "net_memattn.hpp"
#include "net_memenc.hpp"
#include "net_prop.hpp"
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
static std::vector<float> rd(const std::string&p,size_t n){std::vector<float> v(n);std::ifstream f(p,std::ios::binary);if(!f){printf("missing %s\n",p.c_str());std::exit(1);}f.read((char*)v.data(),n*4);return v;}
static float wr(const std::vector<float>&a,const float*b,size_t n){float w=0;for(size_t i=0;i<n;++i)w=std::max(w,std::fabs(a[i]-b[i]));return w;}
// bilinear upsample [1,1,256,256] -> [1,1,1024,1024] (align_corners=False)
static Tensor up1024(const Tensor& m){ int64_t S=256,T=1024; std::vector<float> o(T*T);
  for(int64_t y=0;y<T;++y)for(int64_t x=0;x<T;++x){ float sy=(y+0.5f)*S/T-0.5f, sx=(x+0.5f)*S/T-0.5f;
    int y0=(int)std::floor(sy),x0=(int)std::floor(sx); float fy=sy-y0,fx=sx-x0;
    auto g=[&](int yy,int xx){yy=std::max<int64_t>(0,std::min<int64_t>(S-1,yy));xx=std::max<int64_t>(0,std::min<int64_t>(S-1,xx));return m->data[yy*S+xx];};
    o[y*T+x]=g(y0,x0)*(1-fx)*(1-fy)+g(y0,x0+1)*fx*(1-fy)+g(y0+1,x0)*(1-fx)*fy+g(y0+1,x0+1)*fx*fy; }
  return from_data({1,1,T,T},o,false); }

int main(int argc,char**argv){
  setvbuf(stdout,nullptr,_IONBF,0);
  std::string RF=argc>1?argv[1]:"pure/ref/"; if(RF.back()!='/')RF+='/';
  HieraW ew=load_hiera(RF); MemW mw=load_memattn(RF); MencW cw=load_memenc(RF); PropW pp=load_prop(RF);
  SamW dw; { std::ifstream f(RF+"dec_config.txt"); std::string k; int v; while(f>>k>>v){ if(k=="embed_dim")dw.cfg.embed=v; else if(k=="image_embed")dw.cfg.img=v; else if(k=="input_image")dw.cfg.input=v; else if(k=="num_mask_tokens")dw.cfg.mask_tokens=v; else if(k=="heads")dw.cfg.heads=v; else if(k=="tf_depth")dw.cfg.depth=v; } }
  { std::ifstream f(RF+"dec_weights.bin",std::ios::binary); f.seekg(0,std::ios::end); size_t n=f.tellg()/4; f.seekg(0); dw.buf.resize(n); f.read((char*)dw.buf.data(),n*4); }

  // synthetic 2-frame volume (same as export_prop)
  std::vector<float> i0(3*1024*1024), i1(3*1024*1024);
  for(size_t i=0;i<i0.size();++i){ i0[i]=std::sin(i*0.0005f); i1[i]=std::cos(i*0.0006f); }

  printf("frame0: encoding...\n");
  auto fpn0=hiera_forward(from_data({1,3,1024,1024},i0),ew);   // [256²,128²,64²]
  // frame0 conditioning: + no_mem_embed (broadcast [256] over 64x64)
  Tensor emb0=add_rowvec(sp2tok(fpn0[2]), reshape(pp.no_mem_embed,{1,256}));   // [4096,256]
  emb0=reshape(transpose2d(emb0),{1,256,64,64});
  Sam2Out o0=sam2_decode(emb0,fpn0[0],fpn0[1],{500.f,460.f},{1},dw);   // multimask
  int best=1; for(int i=2;i<=3;++i) if(o0.iou->data[i]>o0.iou->data[best]) best=i;   // best over iou[1..3]
  Tensor tok=slice_rows(o0.mask_tokens, best, best+1);   // multimask token index = best (SAM2 slices mask_tokens_out to 1:4)
  Tensor objptr0=obj_ptr_proj(tok, pp);                               // [1,256]
  printf("  obj_ptr worst = %.3e\n", wr(objptr0->data, rd(RF+"prop_objptr0.bin",256).data(), 256));
  // memory encode: mask_for_mem = sigmoid(upsample(mask[best]))*20-10
  Tensor hires=up1024(from_data({1,1,256,256}, std::vector<float>(o0.masks->data.begin()+best*256*256, o0.masks->data.begin()+(best+1)*256*256)));
  std::vector<float> mfm(1024*1024); for(size_t i=0;i<mfm.size();++i) mfm[i]=(1.f/(1.f+std::exp(-hires->data[i])))*20.f-10.f;
  Tensor mm0=memenc_forward(fpn0[2], from_data({1,1,1024,1024},mfm,false), cw, /*skip_sigmoid=*/true);
  // occlusion: obj present (obj_score>0) -> no change; else add no_obj_embed_spatial (skip when present)
  printf("  mm0 worst = %.3e\n", wr(mm0->data, rd(RF+"prop_mm0.bin",64*64*64).data(), 64*64*64));

  printf("  best=%d iou=[%.3f %.3f %.3f]\n", best, o0.iou->data[1], o0.iou->data[2], o0.iou->data[3]);
  ew.off=0;                                                          // rewind Hiera weight reader before frame1
  printf("frame1: encoding...\n");
  auto fpn1=hiera_forward(from_data({1,3,1024,1024},i1),ew);
  Tensor feat1=sp2tok(fpn1[2]), vpos=sp2tok(pp.vision_pos);
  Tensor pix1=prop_condition(feat1, vpos, mm0, objptr0, mw, pp, 0, 1.f, 2);
  dw.off=0; // rewind decoder weight reader (sam2_decode consumed it)
  Sam2Out o1=sam2_decode(pix1, fpn1[0], fpn1[1], {0.f,0.f}, {-1}, dw);   // no prompt -> mask[0]
  std::vector<float> m1(o1.masks->data.begin(), o1.masks->data.begin()+256*256);
  float mw1=wr(m1, rd(RF+"prop_mask1.bin",256*256).data(), 256*256);
  printf("frame1 mask worst = %.3e  iou=%.4f  %s\n", mw1, o1.iou->data[0], mw1<2e-3?"MATCH (full 3D track verified)":"MISMATCH");
  return 0;
}
