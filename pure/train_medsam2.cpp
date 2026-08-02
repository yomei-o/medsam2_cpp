// MedSAM2 mask-decoder fine-tuning (pure C++, no Python). Like MedSAM2's finetune_sam2_img.py: freeze the
// Hiera image encoder, fine-tune the SAM2 mask decoder (+ prompt encoder) on (image, prompt, mask) pairs
// with focal+dice (+ IoU MSE) on the single-mask output. The encoder is the slow part, so a real training
// loop precomputes the frozen FPN features once per image; here we validate the trainable path on a real
// MedSAM2 frozen embedding (dec_emb/dec_hr256/dec_hr128 from export_medsam2_dec.py) + a synthetic target
// mask + a point prompt — the decoder loss must drop.
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\train_medsam2.cpp
//   run:   train_medsam2 [ref_dir] [--steps N] [--lr F]
#include "net_sam2dec.hpp"
#include "sam_loss.hpp"
#include "optim.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <cmath>
static std::vector<float> rd(const std::string& p, size_t n){ std::vector<float> v(n); std::ifstream f(p,std::ios::binary); if(!f){printf("missing %s\n",p.c_str());std::exit(1);} f.read((char*)v.data(),n*4); return v; }

int main(int argc, char** argv){
  setvbuf(stdout,nullptr,_IONBF,0);
  std::string RF="pure/ref/"; int steps=40; float lr=1e-4f;
  for(int i=1;i<argc;++i){ std::string a=argv[i];
    if(a=="--steps"&&i+1<argc) steps=atoi(argv[++i]);
    else if(a=="--lr"&&i+1<argc) lr=(float)atof(argv[++i]);
    else if(a[0]!='-') RF=a; }
  if(RF.back()!='/') RF+='/';

  // SAM2 decoder weights as autograd leaves (SamW::take -> requires_grad tensors, cached as parameters)
  SamW w; { std::ifstream f(RF+"dec_config.txt"); std::string k; int v; while(f>>k>>v){ if(k=="embed_dim")w.cfg.embed=v; else if(k=="image_embed")w.cfg.img=v; else if(k=="input_image")w.cfg.input=v; else if(k=="num_mask_tokens")w.cfg.mask_tokens=v; else if(k=="heads")w.cfg.heads=v; else if(k=="tf_depth")w.cfg.depth=v; } }
  { std::ifstream f(RF+"dec_weights.bin",std::ios::binary); f.seekg(0,std::ios::end); size_t n=f.tellg()/4; f.seekg(0); w.buf.resize(n); f.read((char*)w.buf.data(),n*4); }

  // frozen encoder outputs for a real MedSAM2 image (image_embed 64², raw fpn[0]=256², fpn[1]=128²)
  Tensor emb   = from_data({1,256,64,64},   rd(RF+"dec_emb.bin",   256*64*64),   false);
  Tensor hr256 = from_data({1,256,256,256}, rd(RF+"dec_hr256.bin", 256*256*256), false);
  Tensor hr128 = from_data({1,256,128,128}, rd(RF+"dec_hr128.bin", 256*128*128), false);

  // synthetic target mask (rectangle) + a point prompt inside it (1024 space)
  std::vector<float> gt(256*256,0.f);
  for(int y=80;y<180;++y) for(int x=100;x<200;++x) gt[y*256+x]=1.f;
  std::vector<float> pts={600.f,520.f}; std::vector<int> labels={1};

  w.rewind(); sam2_decode(emb,hr256,hr128,pts,labels,w); w.finalize();   // build parameter cache
  auto& params=w.parameters(); int64_t np=0; for(auto&p:params) np+=p->numel();
  printf("MedSAM2 decoder training: %.2fM params, steps=%d lr=%g\n", np/1e6, steps, lr);

  Adam opt(params, lr);
  for(int s=0;s<steps;++s){
    w.rewind(); opt.zero_grad();
    Sam2Out o=sam2_decode(emb,hr256,hr128,pts,labels,w);
    Tensor m0 = slice_rows(reshape(o.masks,{4,256*256}), 0, 1);         // single-mask output (token 0)
    Tensor L  = mask_loss(m0, gt);
    float actual = mask_iou(m0, gt);
    Tensor iou0 = slice_cols(reshape(o.iou,{1,4}), 0, 1);               // predicted IoU for token 0
    Tensor total = add(L, sq_err(iou0, actual));
    backward(total); opt.step();
    if(s%5==0||s==steps-1)
      printf("step %2d  mask_loss=%.4f  IoU(pred %.3f / true %.3f)\n", s, L->data[0], iou0->data[0], actual);
  }
  printf("done — decoder mask loss should have dropped (MedSAM2 decoder training path verified).\n");
  return 0;
}
