// Dump Tesseract LSTM internal activations using private API.
// Links against libtesseract to access internal Network classes.
//
// Build:
//   g++ -O2 -I/mnt/volume1/CrispEmbed/tmp/tesseract-src/src \
//       -I/mnt/volume1/CrispEmbed/tmp/tesseract-src/src/ccutil \
//       -I/mnt/volume1/CrispEmbed/tmp/tesseract-src/src/lstm \
//       -I/mnt/volume1/CrispEmbed/tmp/tesseract-src/src/ccstruct \
//       -I/mnt/volume1/CrispEmbed/tmp/tesseract-src/src/arch \
//       -o dump_tess_internals dump_tess_internals.cpp \
//       -ltesseract -lleptonica -std=c++17

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <cstdio>
#include <cstring>
#include <fstream>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s image.png [tessdata_dir]\n", argv[0]);
        return 1;
    }
    const char* image_path = argv[1];
    const char* tessdata = argc > 2 ? argv[2] : nullptr;

    PIX* pix = pixRead(image_path);
    if (!pix) { fprintf(stderr, "Cannot read: %s\n", image_path); return 1; }
    PIX* gray = pixConvertTo8(pix, 0);
    pixDestroy(&pix);

    int w = pixGetWidth(gray);
    int h = pixGetHeight(gray);
    fprintf(stderr, "Image: %dx%d\n", w, h);

    // Dump Leptonica pixel values to file for Python comparison
    {
        FILE* f = fopen("tess_pixels.bin", "wb");
        int wpl = pixGetWpl(gray);
        l_uint32* data = pixGetData(gray);
        for (int y = 0; y < h; y++) {
            l_uint32* line = data + y * wpl;
            for (int x = 0; x < w; x++) {
                uint8_t pixel = GET_DATA_BYTE(line, x);
                fwrite(&pixel, 1, 1, f);
            }
        }
        fclose(f);
        fprintf(stderr, "Wrote tess_pixels.bin (%d bytes)\n", w * h);
    }

    // Run Tesseract and get LSTM choice iterator for per-timestep outputs
    tesseract::TessBaseAPI api;
    if (api.Init(tessdata, "eng", tesseract::OEM_LSTM_ONLY)) {
        fprintf(stderr, "Init failed\n");
        pixDestroy(&gray);
        return 1;
    }
    api.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
    api.SetImage(gray);
    api.Recognize(nullptr);

    // Use ChoiceIterator to get per-symbol alternatives with confidences
    tesseract::ResultIterator* ri = api.GetIterator();
    if (ri) {
        int sym_idx = 0;
        do {
            const char* symbol = ri->GetUTF8Text(tesseract::RIL_SYMBOL);
            float conf = ri->Confidence(tesseract::RIL_SYMBOL);
            if (symbol) {
                // Get bounding box
                int x1, y1, x2, y2;
                ri->BoundingBox(tesseract::RIL_SYMBOL, &x1, &y1, &x2, &y2);
                printf("sym[%d]: '%s' conf=%.2f bbox=(%d,%d)-(%d,%d)\n",
                       sym_idx, symbol, conf, x1, y1, x2, y2);

                // Get choice alternatives
                tesseract::ChoiceIterator ci(*ri);
                int alt = 0;
                do {
                    const char* alt_text = ci.GetUTF8Text();
                    float alt_conf = ci.Confidence();
                    if (alt_text && alt < 5) {
                        printf("  alt[%d]: '%s' conf=%.4f\n", alt, alt_text, alt_conf);
                    }
                    alt++;
                } while (ci.Next());

                delete[] symbol;
                sym_idx++;
            }
        } while (ri->Next(tesseract::RIL_SYMBOL));
        delete ri;
    }

    pixDestroy(&gray);
    api.End();
    return 0;
}
