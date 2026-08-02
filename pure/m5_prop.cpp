// Propagation wiring parity: pure-C++ memory-conditioned feature (frame t attends to frame 0's memory)
// vs SAM2 _prepare_memory_conditioned_features. Validates the full 3D/video conditioning path.
#include "net_prop.hpp"
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
static std::vector<float> rd(const std::string& p, size_t n){ std::vector<float> v(n); std::ifstream f(p,std::ios::binary); if(!f){printf("missing %s\n",p.c_str());std::exit(1);} f.read((char*)v.data(),n*4); return v; }
static float wr(const std::vector<float>& a, const float* b, size_t n){ float w=0; for(size_t i=0;i<n;++i) w=std::max(w,std::fabs(a[i]-b[i])); return w; }
int main(int argc, char** argv){
  std::string RF=argc>1?argv[1]:"pure/ref/"; if(RF.back()!='/')RF+='/';
  MemW mw=load_memattn(RF); PropW p=load_prop(RF);
  Tensor feat1=from_data({4096,256},rd(RF+"prop_feat1.bin",4096*256));
  Tensor pos1=from_data({4096,256},rd(RF+"prop_pos1.bin",4096*256));
  Tensor mm0=from_data({1,64,64,64},rd(RF+"prop_mm0.bin",64*64*64));
  Tensor objptr0=from_data({1,256},rd(RF+"prop_objptr0.bin",256));
  Tensor pix1=prop_condition(feat1,pos1,mm0,objptr0,mw,p,/*t_pos=*/0,/*dt=*/1.f,/*num_frames=*/2);
  auto ref=rd(RF+"prop_pix1.bin",256*64*64); float w=wr(pix1->data,ref.data(),256*64*64);
  double mr=0; for(float v:ref)mr+=std::fabs(v); mr/=ref.size();
  printf("propagation conditioned feat worst = %.3e  (mean|ref| %.3f)  %s\n", w, mr, w<2e-3?"MATCH":"MISMATCH");
  return 0;
}
