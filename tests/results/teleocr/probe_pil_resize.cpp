#include "image_preprocess.h"
#include <cstdio>
#include <vector>
int main(int argc,char**argv) {
  if(argc!=3)return 1;
  const int w=941,h=205;
  std::vector<unsigned char> p(w*h*3);
  std::FILE *f=std::fopen(argv[1],"rb");if(!f)return 2;std::fread(p.data(),1,p.size(),f);std::fclose(f);
  const int rh=196,rw=952;
  std::vector<float> out(rh*rw*3);
  image_preproc::resize_bicubic_pil_u8_hwc(p.data(),h,w,out.data(),rh,rw,3);
  f=std::fopen(argv[2],"wb"); if(!f)return 3;
  std::fwrite(out.data(),sizeof(float),out.size(),f);std::fclose(f);
}
