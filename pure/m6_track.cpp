// Track frame-t decode parity: pure-C++ SAM2 decode on the memory-conditioned feature (no prompt =
// dummy point label -1) vs PyTorch MedSAM2 frame-1 propagated mask. This is the last track-driver piece.
#include "net_sam2dec.hpp"
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
static std::vector<float> rd(const std::string&p,size_t n){std::vector<float> v(n);std::ifstream f(p,std::ios::binary);if(!f){printf("missing %s\n",p.c_str());std::exit(1);}f.read((char*)v.data(),n*4);return v;}
static float wr(const std::vector<float>&a,const float*b,size_t n){float w=0;for(size_t i=0;i<n;++i)w=std::max(w,std::fabs(a[i]-b[i]));return w;}
int main(int argc,char**argv){
  std::string RF=argc>1?argv[1]:"pure/ref/"; if(RF.back()!='/')RF+='/';
  SamW dw; { std::ifstream f(RF+"dec_config.txt"); std::string k; int v; while(f>>k>>v){ if(k=="embed_dim")dw.cfg.embed=v; else if(k=="image_embed")dw.cfg.img=v; else if(k=="input_image")dw.cfg.input=v; else if(k=="num_mask_tokens")dw.cfg.mask_tokens=v; else if(k=="heads")dw.cfg.heads=v; else if(k=="tf_depth")dw.cfg.depth=v; } }
  { std::ifstream f(RF+"dec_weights.bin",std::ios::binary); f.seekg(0,std::ios::end); size_t n=f.tellg()/4; f.seekg(0); dw.buf.resize(n); f.read((char*)dw.buf.data(),n*4); }
  Tensor pix1=from_data({1,256,64,64},rd(RF+"prop_pix1.bin",256*64*64));
  Tensor hr256=from_data({1,256,256,256},rd(RF+"prop_hr256_1.bin",256*256*256));
  Tensor hr128=from_data({1,256,128,128},rd(RF+"prop_hr128_1.bin",256*128*128));
  std::vector<float> pts={0,0}; std::vector<int> lab={-1};            // no prompt -> dummy point label -1
  Sam2Out o=sam2_decode(pix1,hr256,hr128,pts,lab,dw);
  auto mr=rd(RF+"prop_mask1.bin",256*256), ir=rd(RF+"prop_iou1.bin",1);
  std::vector<float> m0(o.masks->data.begin(), o.masks->data.begin()+256*256);   // mask[0] (single output)
  printf("frame1 mask worst = %.3e   iou pure=%.4f ref=%.4f   %s\n", wr(m0,mr.data(),256*256), o.iou->data[0], ir[0],
         wr(m0,mr.data(),256*256)<2e-3?"MATCH":"MISMATCH");
  return 0;
}
