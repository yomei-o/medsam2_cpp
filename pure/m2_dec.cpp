// SAM2 decoder parity: pure-C++ prompt encoder + mask decoder vs PyTorch MedSAM2 (real encoder feats).
#include "net_sam2dec.hpp"
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
static std::vector<float> rd(const std::string&p,size_t n){std::vector<float> v(n);std::ifstream f(p,std::ios::binary);if(!f){printf("missing %s\n",p.c_str());std::exit(1);}f.read((char*)v.data(),n*4);return v;}
static float wr(const std::vector<float>&a,const float*b,size_t n){float w=0;for(size_t i=0;i<n;++i)w=std::max(w,std::fabs(a[i]-b[i]));return w;}
int main(int argc,char**argv){
  std::string RF=argc>1?argv[1]:"pure/ref/"; if(RF.back()!='/')RF+='/';
  SamW w; { std::ifstream f(RF+"dec_config.txt"); std::string k; int v; while(f>>k>>v){ if(k=="embed_dim")w.cfg.embed=v; else if(k=="image_embed")w.cfg.img=v; else if(k=="input_image")w.cfg.input=v; else if(k=="num_mask_tokens")w.cfg.mask_tokens=v; else if(k=="heads")w.cfg.heads=v; else if(k=="tf_depth")w.cfg.depth=v; } }
  { std::ifstream f(RF+"dec_weights.bin",std::ios::binary); f.seekg(0,std::ios::end); size_t n=f.tellg()/4; f.seekg(0); w.buf.resize(n); f.read((char*)w.buf.data(),n*4); }
  Tensor emb=from_data({1,256,64,64}, rd(RF+"dec_emb.bin",256*64*64));
  Tensor hr256=from_data({1,256,256,256}, rd(RF+"dec_hr256.bin",256*256*256));
  Tensor hr128=from_data({1,256,128,128}, rd(RF+"dec_hr128.bin",256*128*128));
  std::vector<float> pts={500,460}; std::vector<int> lab={1};
  Sam2Out o=sam2_decode(emb,hr256,hr128,pts,lab,w);
  auto ir=rd(RF+"dec_iou.bin",3), mr=rd(RF+"dec_masks.bin",3*256*256), obr=rd(RF+"dec_obj.bin",1);
  std::vector<float> iou3={o.iou->data[1],o.iou->data[2],o.iou->data[3]};
  std::vector<float> m3(o.masks->data.begin()+256*256, o.masks->data.begin()+4*256*256);
  printf("iou pure=[%.4f %.4f %.4f] ref=[%.4f %.4f %.4f] worst=%.2e\n", iou3[0],iou3[1],iou3[2],ir[0],ir[1],ir[2],wr(iou3,ir.data(),3));
  printf("obj pure=%.4f ref=%.4f\n", o.obj->data[0], obr[0]);
  float mw=wr(m3,mr.data(),3*256*256);
  printf("masks worst=%.3e  %s\n", mw, mw<2e-3?"MATCH":"MISMATCH");
  return 0;
}
