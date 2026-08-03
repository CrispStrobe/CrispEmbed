// image_out.cpp — PNG encoding, provenance metadata, and optional C2PA signing.
//
// The c2pa header and library are confined to this translation unit, so a build
// without them differs in exactly one function's body.

#include "core/image_out.h"

// Single definition for the whole project: ocr_orchestrator.cpp used to own
// it, but this layer is lower and every target links it. Keep stdio in —
// ocr_orchestrator writes crops by path and a test externs stbi_write_png.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../ggml/examples/stb_image_write.h"

#ifdef CRISPEMBED_HAVE_C2PA
#include "c2pa.h"
#endif

#include <fstream>
#include <sstream>

namespace core_imgout {

namespace {

void collect(void * ctx, void * data, int size) {
    auto * s = static_cast<std::string *>(ctx);
    s->append(static_cast<const char *>(data), (size_t)size);
}

std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream o;
    o << f.rdbuf();
    return o.str();
}

// What the iTXt says. Deliberately the same vocabulary as the C2PA assertion
// below, so the two levels of marking cannot drift apart.
std::string provenance_text(const char * engine) {
    std::string t = "generated=true\n";
    t += "software=CrispEmbed\n";
    if (engine && *engine) {
        t += "engine=";
        t += engine;
        t += "\n";
    }
    // IPTC DigitalSourceType. algorithmicallyEnhanced, NOT trainedAlgorithmicMedia:
    // the input is a real capture that we enhanced. Claiming the stronger term
    // would assert the image is wholly synthetic, which is false and a worse
    // position than saying nothing.
    t += "digitalSourceType=http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicallyEnhanced\n";
    t += "note=AI-processed image. Not an authentic record of the original; restored or "
         "upscaled detail is a plausible completion, not recovered information.\n";
    t += "policy=https://github.com/CrispStrobe/CrispEmbed/blob/main/POLICY.md\n";
    return t;
}

#ifdef CRISPEMBED_HAVE_C2PA
struct membuf {
    std::string * s;
    long pos;
};
intptr_t mem_read(StreamContext * c, uint8_t * d, intptr_t n) {
    auto * m = reinterpret_cast<membuf *>(c);
    long avail = (long)m->s->size() - m->pos;
    if (avail <= 0) return 0;
    long k = n < avail ? n : avail;
    std::memcpy(d, m->s->data() + m->pos, (size_t)k);
    m->pos += k;
    return k;
}
intptr_t mem_seek(StreamContext * c, intptr_t off, C2paSeekMode mode) {
    auto * m = reinterpret_cast<membuf *>(c);
    long base = mode == Start ? 0 : (mode == Current ? m->pos : (long)m->s->size());
    m->pos = base + off;
    return m->pos;
}
intptr_t mem_write(StreamContext * c, const uint8_t * d, intptr_t n) {
    auto * m = reinterpret_cast<membuf *>(c);
    if ((long)m->s->size() < m->pos + n) m->s->resize((size_t)(m->pos + n));
    std::memcpy(&(*m->s)[(size_t)m->pos], d, (size_t)n);
    m->pos += n;
    return n;
}
intptr_t mem_flush(StreamContext *) {
    return 0;
}
#endif

} // namespace

bool c2pa_sign_png(std::string & png, const char * engine) {
#ifndef CRISPEMBED_HAVE_C2PA
    (void)png;
    (void)engine;
    return false;
#else
    if (!c2pa_configured()) return false;
    const std::string cert = read_file(std::getenv("CRISPEMBED_C2PA_CERT"));
    const std::string key = read_file(std::getenv("CRISPEMBED_C2PA_KEY"));
    if (cert.empty() || key.empty()) {
        std::fprintf(stderr, "crispembed: C2PA cert/key unreadable — emitting unsigned (still marked)\n");
        return false;
    }

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    C2paSignerInfo info;
    info.alg = "es256";
    info.sign_cert = cert.c_str();
    info.private_key = key.c_str();
    info.ta_url = nullptr;
    C2paSigner * signer = c2pa_signer_from_info(&info);
    if (!signer) {
        // The common cause is a self-signed certificate, which c2pa-rs refuses.
        // Say so rather than leaving the caller to guess.
        std::fprintf(stderr, "crispembed: C2PA signer init failed: %s\n", c2pa_error() ? c2pa_error() : "unknown");
        std::fprintf(stderr, "            a self-signed cert is rejected; use a leaf+CA chain "
                             "(see scripts/make-c2pa-cert.sh)\n");
        return false;
    }

    std::string manifest = "{\n"
                           "  \"claim_generator\": \"CrispEmbed\",\n"
                           "  \"claim_generator_info\": [{ \"name\": \"CrispEmbed\" }],\n"
                           "  \"assertions\": [{\n"
                           "    \"label\": \"c2pa.actions\",\n"
                           "    \"data\": { \"actions\": [{\n"
                           "      \"action\": \"c2pa.edited\",\n"
                           "      \"digitalSourceType\": \"http://cv.iptc.org/newscodes/"
                           "digitalsourcetype/algorithmicallyEnhanced\",\n"
                           "      \"softwareAgent\": \"CrispEmbed ";
    manifest += (engine && *engine) ? engine : "image";
    manifest += "\"\n    }]}\n  }]\n}";

    C2paBuilder * builder = c2pa_builder_from_json(manifest.c_str());
    if (!builder) {
        std::fprintf(stderr, "crispembed: C2PA builder init failed: %s\n", c2pa_error() ? c2pa_error() : "unknown");
        c2pa_signer_free(signer);
        return false;
    }

    membuf src{ &png, 0 };
    std::string out;
    membuf dst{ &out, 0 };
    C2paStream * ss =
        c2pa_create_stream(reinterpret_cast<StreamContext *>(&src), mem_read, mem_seek, mem_write, mem_flush);
    C2paStream * ds =
        c2pa_create_stream(reinterpret_cast<StreamContext *>(&dst), mem_read, mem_seek, mem_write, mem_flush);
    const unsigned char * mb = nullptr;
    int64_t rc = c2pa_builder_sign(builder, "image/png", ss, ds, signer, &mb);
    if (mb) c2pa_manifest_bytes_free(mb);
    c2pa_release_stream(ss);
    c2pa_release_stream(ds);
    c2pa_builder_free(builder);
    c2pa_signer_free(signer);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

    if (rc < 0) {
        std::fprintf(stderr, "crispembed: C2PA signing failed: %s\n", c2pa_error() ? c2pa_error() : "unknown");
        return false;
    }
    png.swap(out);
    return true;
#endif
}

bool emit(std::FILE * out, const uint8_t * data, int w, int h, int comp, const char * engine) {
    if (!out) out = stdout;
    if (!data || w <= 0 || h <= 0 || (comp != 1 && comp != 3)) return false;

    if (want_ppm()) {
        // Historical raw Netpbm, kept for callers that parse it directly.
        std::fprintf(out, "%s\n%s%d %d\n255\n", comp == 3 ? "P6" : "P5", core_prov::netpbm_comment(engine).c_str(), w,
                     h);
        return std::fwrite(data, 1, (size_t)w * h * comp, out) == (size_t)w * h * comp;
    }

    std::string png;
    if (!stbi_write_png_to_func(collect, &png, w, h, comp, data, w * comp)) return false;

    // Mark first, sign second: the signature must cover the metadata, not
    // precede it. Signing then inserting a chunk would invalidate the hash.
    const std::string chunk = png_itxt("CrispEmbed", provenance_text(engine));
    if (!png_insert_itxt(png, chunk)) {
        std::fprintf(stderr, "crispembed: could not attach provenance metadata (unexpected PNG layout)\n");
    }
    c2pa_sign_png(png, engine); // best effort; unsigned output is still marked

    return std::fwrite(png.data(), 1, png.size(), out) == png.size();
}

} // namespace core_imgout
