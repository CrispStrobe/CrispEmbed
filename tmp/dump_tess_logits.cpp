// Dump Tesseract LSTM raw softmax logits for a single line image.
// Compile:
//   g++ -O2 -o dump_tess_logits dump_tess_logits.cpp \
//       $(pkg-config --cflags --libs tesseract lept) -std=c++17
// Usage:
//   ./dump_tess_logits /path/to/image.png [tessdata_dir]

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s image.png [tessdata_dir]\n", argv[0]);
        return 1;
    }
    const char* image_path = argv[1];
    const char* tessdata = argc > 2 ? argv[2] : nullptr;

    // Load image with Leptonica
    PIX* pix = pixRead(image_path);
    if (!pix) {
        fprintf(stderr, "Cannot read image: %s\n", image_path);
        return 1;
    }
    // Convert to 8-bit grayscale
    PIX* gray = pixConvertTo8(pix, 0);
    pixDestroy(&pix);
    int w = pixGetWidth(gray);
    int h = pixGetHeight(gray);
    fprintf(stderr, "Image: %dx%d\n", w, h);

    // Initialize Tesseract
    tesseract::TessBaseAPI api;
    if (api.Init(tessdata, "eng", tesseract::OEM_LSTM_ONLY)) {
        fprintf(stderr, "Tesseract init failed\n");
        pixDestroy(&gray);
        return 1;
    }
    api.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
    api.SetImage(gray);

    // Run recognition
    api.Recognize(nullptr);

    // Get the text result
    char* text = api.GetUTF8Text();
    fprintf(stderr, "Text: '%s'\n", text ? text : "(null)");
    if (text) delete[] text;

    // Get raw LSTM output via ResultIterator
    // We can get per-character confidence, but not raw logits via public API.
    // Let's at least dump the character-level results.
    tesseract::ResultIterator* ri = api.GetIterator();
    if (ri) {
        int idx = 0;
        do {
            const char* symbol = ri->GetUTF8Text(tesseract::RIL_SYMBOL);
            float conf = ri->Confidence(tesseract::RIL_SYMBOL);
            if (symbol) {
                printf("char[%d]: '%s' conf=%.4f\n", idx++, symbol, conf);
                delete[] symbol;
            }
        } while (ri->Next(tesseract::RIL_SYMBOL));
        delete ri;
    }

    // Dump raw pixel values that Tesseract would see (after its preprocessing)
    // to compare normalization
    fprintf(stderr, "\n--- Pixel samples (after Leptonica 8-bit conversion) ---\n");
    int wpl = pixGetWpl(gray);
    l_uint32* data = pixGetData(gray);
    // Middle row
    int mid_y = h / 2;
    l_uint32* line = data + mid_y * wpl;
    fprintf(stderr, "Row %d: ", mid_y);
    for (int x = 0; x < w && x < 20; x++) {
        int pixel = GET_DATA_BYTE(line, x);
        fprintf(stderr, "%d ", pixel);
    }
    fprintf(stderr, "...\n");

    // Check a text pixel
    for (int x = 96; x < 106; x++) {
        for (int y = 10; y < 25; y++) {
            l_uint32* row = data + y * wpl;
            int pixel = GET_DATA_BYTE(row, x);
            if (pixel < 128) {
                fprintf(stderr, "Dark pixel at (%d,%d): value=%d\n", x, y, pixel);
            }
        }
    }

    pixDestroy(&gray);
    api.End();
    return 0;
}
