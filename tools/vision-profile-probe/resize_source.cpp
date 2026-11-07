// Built only inside the pinned upstream checkout; invokes its actual resizer.
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
extern "C" bool crisp_reference_pillow22(const uint8_t*,int,int,int,int,uint8_t*);
int main(int argc,char**argv){try{
 if(argc!=7)throw std::runtime_error("RGB height width output-height output-width prefix required");
 int h=std::stoi(argv[2]),w=std::stoi(argv[3]),oh=std::stoi(argv[4]),ow=std::stoi(argv[5]);
 if(h<1||w<1||oh<1||ow<1||h>4096||w>4096||oh>4096||ow>4096)throw std::runtime_error("bounded shape");
 std::vector<uint8_t>in(size_t(h)*w*3),out(size_t(oh)*ow*3);std::ifstream f(argv[1],std::ios::binary);f.read(reinterpret_cast<char*>(in.data()),in.size());
 if(size_t(f.gcount())!=in.size()||f.peek()!=EOF||!crisp_reference_pillow22(in.data(),h,w,oh,ow,out.data()))throw std::runtime_error("reference resize");
 std::ofstream o(std::string(argv[6])+".rgb",std::ios::binary);o.write(reinterpret_cast<const char*>(out.data()),out.size());if(!o)throw std::runtime_error("output");std::cout<<"REFERENCE_PILLOW22 PASS\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
