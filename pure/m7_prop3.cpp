// Multi-frame memory bank parity: frame2 conditioned on frame0 (cond, t_pos=0) + frame1 (t_pos=6) +
// 2 object pointers, vs SAM2 _prepare_memory_conditioned_features (3-frame case).
#include "net_prop.hpp"
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
static std::vector<float> rd(const std::string&p,size_t n){std::vector<float> v(n);std::ifstream f(p,std::ios::binary);if(!f){printf("missing %s\n",p.c_str());std::exit(1);}f.read((char*)v.data(),n*4);return v;}
int main(int argc,char**argv){
  std::string RF=argc>1?argv[1]:"pure/ref/"; if(RF.back()!='/')RF+='/';
  MemW mw=load_memattn(RF); PropW p=load_prop(RF);
  Tensor feat2=from_data({4096,256},rd(RF+"p3_feat2.bin",4096*256)), pos2=from_data({4096,256},rd(RF+"p3_pos2.bin",4096*256));
  Tensor mm0=from_data({1,64,64,64},rd(RF+"p3_mm0.bin",64*64*64)), mm1=from_data({1,64,64,64},rd(RF+"p3_mm1.bin",64*64*64));
  Tensor op0=from_data({1,256},rd(RF+"p3_optr0.bin",256)), op1=from_data({1,256},rd(RF+"p3_optr1.bin",256));
  // frame2 (num_frames=3): spatial [frame0 t_pos0, frame1 t_pos6]; obj ptrs [(optr0,Δt=2),(optr1,Δt=1)]
  std::vector<std::pair<Tensor,int>> mems = {{mm0,0},{mm1,6}};
  std::vector<std::pair<Tensor,float>> ptrs = {{op0,2.f},{op1,1.f}};
  Tensor pix2=prop_condition_multi(feat2,pos2,mems,ptrs,mw,p,/*num_frames=*/3);
  auto ref=rd(RF+"p3_pix2.bin",256*64*64); float w=0; for(size_t i=0;i<pix2->data.size();++i)w=std::max(w,std::fabs(pix2->data[i]-ref[i]));
  double mr=0; for(float v:ref)mr+=std::fabs(v); mr/=ref.size();
  printf("frame2 multi-frame conditioned worst = %.3e  (mean|ref| %.3f)  %s\n", w, mr, w<2e-3?"MATCH":"MISMATCH");
  return 0;
}
