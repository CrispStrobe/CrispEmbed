// test_posformer_batch.cpp — Batch evaluation for PosFormer.
// Usage: ./test-posformer-batch model.gguf image_dir
// Reads all .bmp files from image_dir, outputs TSV: filename\tresult

#include "posformer_ocr.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <vector>
#include <string>
#include <algorithm>

static bool load_bmp_gray(const char * path, std::vector<float> & gray,
                          int & w, int & h) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;

    unsigned char header[54];
    if (fread(header, 1, 54, f) != 54) { fclose(f); return false; }

    w = *(int*)&header[18];
    h = *(int*)&header[22];
    int bpp = *(short*)&header[28];
    int offset = *(int*)&header[10];
    int abs_h = h < 0 ? -h : h;

    fseek(f, offset, SEEK_SET);
    gray.resize(w * abs_h);

    int row_bytes = ((w * (bpp / 8) + 3) / 4) * 4;
    std::vector<unsigned char> row(row_bytes);

    for (int y = 0; y < abs_h; y++) {
        if (fread(row.data(), 1, row_bytes, f) != (size_t)row_bytes) {
            fclose(f); return false;
        }
        int dy = (h > 0) ? (abs_h - 1 - y) : y;
        for (int x = 0; x < w; x++) {
            if (bpp == 24 || bpp == 32) {
                int b = row[x * (bpp/8)], g = row[x*(bpp/8)+1], r = row[x*(bpp/8)+2];
                gray[dy * w + x] = (0.299f*r + 0.587f*g + 0.114f*b) / 255.0f;
            } else if (bpp == 8) {
                gray[dy * w + x] = row[x] / 255.0f;
            }
        }
    }
    h = abs_h;
    fclose(f);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <image_dir>\n", argv[0]);
        return 1;
    }

    posformer_ocr_context * ctx = posformer_ocr_init(argv[1], 4);
    if (!ctx) { fprintf(stderr, "Failed to load model\n"); return 1; }

    // List BMP files
    std::vector<std::string> files;
    DIR * dir = opendir(argv[2]);
    if (!dir) { fprintf(stderr, "Can't open dir: %s\n", argv[2]); return 1; }
    struct dirent * ent;
    while ((ent = readdir(dir)) != nullptr) {
        std::string name = ent->d_name;
        if (name.size() > 4 && name.substr(name.size()-4) == ".bmp")
            files.push_back(name);
    }
    closedir(dir);
    std::sort(files.begin(), files.end());
    fprintf(stderr, "Found %zu BMP files\n", files.size());

    for (size_t i = 0; i < files.size(); i++) {
        std::string path = std::string(argv[2]) + "/" + files[i];
        int W, H;
        std::vector<float> img;
        if (!load_bmp_gray(path.c_str(), img, W, H)) {
            fprintf(stderr, "SKIP: %s\n", files[i].c_str());
            continue;
        }

        int len = 0;
        const char * result = posformer_ocr_recognize(ctx, img.data(), W, H, &len);
        // Output TSV: filename \t result
        printf("%s\t%s\n", files[i].c_str(), result ? result : "");
        fflush(stdout);

        if ((i + 1) % 50 == 0)
            fprintf(stderr, "Progress: %zu/%zu\n", i + 1, files.size());
    }

    posformer_ocr_free(ctx);
    return 0;
}
