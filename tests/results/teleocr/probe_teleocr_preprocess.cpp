#include "qwen2vl_ocr.h"
#include "image_preprocess.h"
#include "core/clean_exit.h"
#include "stb_image.h"
#include <cstdio>
static int run(int argc,char**argv) {
  if(argc!=3)return 1;
  int w=0,h=0,c=0;unsigned char *p=stbi_load(argv[1],&w,&h,&c,3);if(!p)return 2;
  image_preproc::config cfg;cfg.min_pixels=3136;cfg.max_pixels=12845056;
  cfg.patch_size=14;cfg.temporal_patch_size=2;cfg.merge_size=2;
  for(int i=0;i<3;i++){cfg.mean[i]=image_preproc::kCLIPMean[i];cfg.std[i]=image_preproc::kCLIPStd[i];}
  image_preproc::result out;
  if(!qwen2vl_ocr::preprocess_teleocr(p,h,w,3,cfg,out)){stbi_image_free(p);return 3;}
  stbi_image_free(p);std::FILE*f=std::fopen(argv[2],"wb");if(!f)return 4;
  auto n=std::fwrite(out.patches.data(),sizeof(float),out.patches.size(),f);std::fclose(f);
  std::printf("%dx%d -> %dx%d n=%d row=%d\n",w,h,out.resized_w,out.resized_h,out.n_patches,out.row_dim);
  return n==out.patches.size()?0:5;
}
int main(int argc,char**argv){core_util::clean_exit(run(argc,argv));}
