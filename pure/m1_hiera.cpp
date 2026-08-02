// Hiera-T trunk parity: pure-C++ stage outputs vs PyTorch MedSAM2 image_encoder.trunk.
#include "net_hiera.hpp"
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
static std::vector<float> rd(const std::string&p,size_t n){std::vector<float> v(n);std::ifstream f(p,std::ios::binary);if(!f){printf("missing %s\n",p.c_str());std::exit(1);}f.read((char*)v.data(),n*4);return v;}
static float wr(const std::vector<float>&a,const std::vector<float>&b){float w=0;size_t n=std::min(a.size(),b.size());for(size_t i=0;i<n;++i)w=std::max(w,std::fabs(a[i]-b[i]));return w;}
int main(int argc,char**argv){
  setvbuf(stdout,nullptr,_IONBF,0);
  std::string RF=argc>1?argv[1]:"pure/ref/"; if(RF.back()!='/')RF+='/';
  HieraW w=load_hiera(RF);
  printf("Hiera-T encoding...\n");
  auto fpn=hiera_forward(from_data({1,3,1024,1024},rd(RF+"hiera_in.bin",3*1024*1024)),w);
  const int64_t sc[4]={96,192,384,768}; const int64_t sr[4]={256,128,64,32};
  for(int s=0;s<4;++s){ auto ref=rd(RF+"hiera_stage"+std::to_string(s)+".bin", sc[s]*sr[s]*sr[s]);
    printf("stage%d worst = %.3e  [%lld,%lld^2]\n", s, wr(g_hiera_stages[s]->data,ref), (long long)sc[s],(long long)sr[s]); }
  const int64_t fr[3]={256,128,64};
  for(int i=0;i<3;++i){ auto ref=rd(RF+"hiera_fpn"+std::to_string(i)+".bin", 256*fr[i]*fr[i]);
    float e=wr(fpn[i]->data,ref);
    printf("fpn%d    worst = %.3e  [256,%lld^2]  %s\n", i, e,(long long)fr[i], e<1e-3?"MATCH":"CHECK"); }
  return 0;
}
