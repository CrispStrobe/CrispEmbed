// Model-free renderer harness: the production library already defines these
// symbols in image_preprocess.cpp and core/image_out.cpp. Only the standalone
// artifact test links this translation unit; never add it to crispembed.
#define STB_IMAGE_IMPLEMENTATION
#include "../ggml/examples/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../ggml/examples/stb_image_write.h"
