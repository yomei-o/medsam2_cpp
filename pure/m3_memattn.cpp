// memory_attention parity: pure-C++ RoPE self+cross attn vs PyTorch MedSAM2 memory_attention.
#include "net_memattn.hpp"
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
static std::vector<float> rd(const std::string&p,size_t n){std::vector<float> v(n);std::ifstream f(p,std::ios::binary);if(!f){printf("missing %s\n",p.c_str());std::exit(1);}f.read((char*)v.data(),n*4);return v;}
int main(int argc,char**argv){
  std::string RF=argc>1?argv[1]:"pure/ref/"; if(RF.back()!='/')RF+='/';
  MemW w=load_memattn(RF); int64_t N=4096;
  Tensor curr=from_data({N,256},rd(RF+"mem_curr.bin",N*256)), currpos=from_data({N,256},rd(RF+"mem_currpos.bin",N*256));
  Tensor mem=from_data({N,64},rd(RF+"mem_mem.bin",N*64)), mempos=from_data({N,64},rd(RF+"mem_mempos.bin",N*64));
  Tensor out=memory_attention(curr,currpos,mem,mempos,w);
  auto ref=rd(RF+"mem_out.bin",N*256); float wst=0; for(size_t i=0;i<out->data.size();++i) wst=std::max(wst,std::fabs(out->data[i]-ref[i]));
  double mr=0; for(float v:ref)mr+=std::fabs(v); mr/=ref.size();
  printf("memory_attention worst = %.3e  (mean|ref| %.3f)  %s\n", wst, mr, wst<2e-3?"MATCH":"MISMATCH");
  return 0;
}
