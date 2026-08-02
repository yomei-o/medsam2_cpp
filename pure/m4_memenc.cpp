// memory_encoder parity: pure-C++ vs PyTorch MedSAM2 memory_encoder.
#include "net_memenc.hpp"
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
static std::vector<float> rd(const std::string&p,size_t n){std::vector<float> v(n);std::ifstream f(p,std::ios::binary);if(!f){printf("missing %s\n",p.c_str());std::exit(1);}f.read((char*)v.data(),n*4);return v;}
int main(int argc,char**argv){
  std::string RF=argc>1?argv[1]:"pure/ref/"; if(RF.back()!='/')RF+='/';
  MencW w=load_memenc(RF);
  Tensor pix=from_data({1,256,64,64},rd(RF+"menc_pix.bin",256*64*64));
  Tensor mk=from_data({1,1,1024,1024},rd(RF+"menc_masks.bin",1024*1024));
  Tensor out=memenc_forward(pix,mk,w);
  auto ref=rd(RF+"menc_out.bin",64*64*64); float wst=0; for(size_t i=0;i<out->data.size();++i)wst=std::max(wst,std::fabs(out->data[i]-ref[i]));
  double mr=0; for(float v:ref)mr+=std::fabs(v); mr/=ref.size();
  printf("memory_encoder worst = %.3e  (mean|ref| %.3f)  %s\n", wst, mr, wst<2e-3?"MATCH":"MISMATCH");
  return 0;
}
