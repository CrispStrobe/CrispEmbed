// Contract test for the native Camel Morph analyzer C ABI (src/arabic_morph.h).
//
// Uses a small synthetic ALMOR database (no CC BY data) and expected outputs
// produced by camel_tools 1.6.0 (MorphologyDB(path, 'a'), Analyzer(db,
// backoff)) with json.dumps(..., ensure_ascii=False, separators=(',', ':')).
// In the NOAN_PROP case the three *_logprob keys, which merge_features appends
// from a frozenset (PYTHONHASHSEED-dependent order), are listed in camel_tools'
// source order, as the port emits them.
// Full-database exactness against camel_tools is recorded in
// docs/validation/camel-morph-msa-20261119/README.md.
//
// The binary format is checked on the same database converted by
// tools/arabic_morph/gen_bin_fixture.py (arabic_morph_bin_fixture.h), against
// the same camel_tools outputs projected onto the fixture's profile; the
// compact ABI is checked against the JSON ABI for both formats.
#include "core/clean_exit.h"
#include "arabic_morph.h"
#include "arabic_morph_bin_fixture.h"
#include "core/json.h"
#include "core/sha256.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

const char * const k_db = R"DB(
###DEFINES###
DEFINE diac diac:*open*
DEFINE lex lex:*open*
DEFINE caphi caphi:*open*
DEFINE gloss gloss:*open*
DEFINE bw bw:*open*
DEFINE pos pos:noun pos:digit pos:punc pos:latin pos:noun_prop
DEFINE prc2 prc2:0 prc2:wa_conj
DEFINE gen gen:m gen:f gen:-
DEFINE num num:s num:-
DEFINE form_gen form_gen:m form_gen:f
DEFINE form_num form_num:s
DEFINE pattern pattern:*open*
DEFINE source source:lex source:spvar source:backoff
DEFINE stem stem:*open*
DEFINE stemcat stemcat:*open*
DEFINE stemgloss stemgloss:*open*
DEFINE d3tok d3tok:*open*
DEFINE pos_logprob pos_logprob:*open*
DEFINE lex_logprob lex_logprob:*open*
DEFINE pos_lex_logprob pos_lex_logprob:*open*
###DEFAULTS###
DEFAULT pos:digit diac:* lex:* caphi:* gloss:* bw:* pos:digit prc2:0 gen:- num:- form_gen:m form_num:s pattern:* source:lex stem:* stemcat:* stemgloss:* d3tok:* pos_logprob:* lex_logprob:* pos_lex_logprob:*
DEFAULT pos:punc diac:* lex:* caphi:* gloss:* bw:* pos:punc prc2:0 gen:m num:s form_gen:m form_num:s pattern:* source:lex stem:* stemcat:* stemgloss:* d3tok:* pos_logprob:* lex_logprob:* pos_lex_logprob:*
DEFAULT pos:latin diac:* lex:* caphi:* gloss:* bw:* pos:latin prc2:0 gen:m num:s form_gen:m form_num:s pattern:* source:lex stem:* stemcat:* stemgloss:* d3tok:* pos_logprob:* lex_logprob:* pos_lex_logprob:*
###ORDER###
ORDER diac lex caphi gloss bw pos prc2 gen num form_gen form_num pattern source stem stemcat stemgloss d3tok pos_logprob lex_logprob pos_lex_logprob
###TOKENIZATIONS###
TOKENIZATION d3tok
###STEMBACKOFF###
STEMBACKOFF ALL XN
STEMBACKOFF PROP XN
###PREFIXES###
	P0	diac: caphi: gloss: bw: prc2:0 d3tok:
و	P1	diac:وَ caphi:w_a gloss:and bw:وَ/CONJ prc2:wa_conj d3tok:وَ+_
###SUFFIXES###
	S0	diac: caphi: gloss: bw: d3tok:
ة	S1	diac:َة caphi:a_p gloss:[fem.sg.] bw:َة/NSUFF_FEM_SG gen:f d3tok:َة
###STEMS###
كتاب	X1	diac:كِتَاب lex:كِتَاب_1 caphi:k_i_t_aa_b gloss:book bw:كِتَاب/NOUN pos:noun gen:m num:s pattern:1ِ2َا3 lex_logprob:-8.084950 pos_lex_logprob:-8.084950
كاتب	X2	diac:كاتِب lex:كاتِب-u caphi:k_aa_t_i_b gloss:writer bw:كاتِب/NOUN pos:noun gen:- form_gen:m num:s lex_logprob:-99 pos_lex_logprob:-11.5
NOAN	XN	diac:NOAN lex:NOAN gloss:NOAN bw:NOAN/NOUN_PROP pos:noun_prop gen:m num:s
###TABLE AB###
P0 X1
P1 X1
P0 X2
P1 X2
P0 XN
P1 XN
###TABLE BC###
X1 S0
X2 S0
X2 S1
XN S0
###TABLE AC###
P0 S0
P1 S0
P0 S1
P1 S1
)DB";

struct expected_case {
    int backoff;
    const char * word;
    const char * json;
};

const expected_case k_cases[] = {
    { 0, "وكتاب",
      R"J([{"diac":"وَكِتَاب","lex":"كِتَاب","caphi":"w_a_k_i_t_aa_b","gloss":"and+book","bw":"وَ/CONJ+كِتَاب/NOUN","pos":"noun","gen":"m","num":"s","pattern":"وَ1ِ2َا3","lex_logprob":-8.08495,"pos_lex_logprob":-8.08495,"d3tok":"وَ+_كِتَاب","stem":"كِتَاب","stemgloss":"book","pos_logprob":-99.0,"stemcat":"X1"}])J" },
    { 0, "وَكِتَابٌ",
      R"J([{"diac":"وَكِتَاب","lex":"كِتَاب","caphi":"w_a_k_i_t_aa_b","gloss":"and+book","bw":"وَ/CONJ+كِتَاب/NOUN","pos":"noun","gen":"m","num":"s","pattern":"وَ1ِ2َا3","lex_logprob":-8.08495,"pos_lex_logprob":-8.08495,"d3tok":"وَ+_كِتَاب","stem":"كِتَاب","stemgloss":"book","pos_logprob":-99.0,"stemcat":"X1"}])J" },
    { 0, "2024",
      R"J([{"pos":"digit","diac":"2024","lex":"2024","caphi":"DIGIT","gloss":"2024","bw":"2024/NOUN_NUM","prc2":"0","gen":"m","num":"s","form_gen":"m","form_num":"s","pattern":"DIGIT","source":"digit","stem":"2024","stemcat":null,"stemgloss":"2024","d3tok":"2024","pos_logprob":-99.0,"lex_logprob":-99.0,"pos_lex_logprob":-99.0}])J" },
    { 0, "«",
      R"J([{"pos":"punc","diac":"«","lex":"«","caphi":"PUNC","gloss":"«","bw":"«/PUNC","prc2":"0","gen":"m","num":"s","form_gen":"m","form_num":"s","pattern":"PUNC","source":"punc","stem":"«","stemcat":null,"stemgloss":"«","d3tok":"«","pos_logprob":-99.0,"lex_logprob":-99.0,"pos_lex_logprob":-99.0}])J" },
    { 0, "abc",
      R"J([{"pos":"foreign","diac":"abc","lex":"abc","caphi":"FOREIGN","gloss":"abc","bw":"abc/FOREIGN","prc2":"0","gen":"m","num":"s","form_gen":"m","form_num":"s","pattern":"FOREIGN","source":"foreign","stem":"abc","stemcat":null,"stemgloss":"abc","d3tok":"abc","pos_logprob":-99.0,"lex_logprob":-99.0,"pos_lex_logprob":-99.0}])J" },
    { 0, "زيد", R"J([])J" },
    { 0, "  ", R"J([])J" },
    { 2, "زيد",
      R"J([{"diac":"زيد","lex":"زيد","gloss":"NOAN","bw":"زيد/NOUN_PROP","pos":"noun_prop","gen":"m","num":"s","caphi":"z_y_d","pattern":"backoff","d3tok":"زيد","stem":"زيد","stemgloss":"NOAN","pos_logprob":-99.0,"lex_logprob":-99.0,"pos_lex_logprob":-99.0,"stemcat":"XN","source":"backoff"}])J" },
};

std::string temp_path(const char * name) {
    const char * dir = std::getenv("TMPDIR");
    std::string p = dir && *dir ? dir : "/tmp";
    if (p.back() != '/') p += '/';
    return p + name;
}

bool write_file(const std::string & path, const std::string & bytes) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
    return std::fclose(f) == 0 && ok;
}

std::string analyze(crispembed_ar_morph * m, const std::string & w, int * rc_out = nullptr) {
    char * json = nullptr;
    size_t len = 0;
    char err[256];
    const int rc = crispembed_ar_morph_analyze(m, w.data(), w.size(), &json, &len, err, sizeof err);
    if (rc_out) *rc_out = rc;
    if (rc != CRISPEMBED_AR_MORPH_OK) return std::string("<error ") + std::to_string(rc) + ">";
    std::string out(json, len);
    crispembed_ar_morph_string_free(json);
    return out;
}

void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}

uint32_t rd32(const unsigned char * p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

uint32_t rd16(const unsigned char * p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8);
}

std::vector<std::string> info_keys(const char * info) {
    const std::string s(info);
    std::vector<std::string> keys;
    size_t i = s.find("\"keys\":[");
    if (i == std::string::npos) return keys;
    for (i += 8; i < s.size() && s[i] != ']';) {
        if (s[i] != '"') {
            ++i;
            continue;
        }
        std::string k;
        size_t end = 0;
        core_json::json_decode_string(s, i, k, end);
        keys.push_back(k);
        i = end;
    }
    return keys;
}

// The compact results of `words`, re-rendered as the JSON ABI renders them
// ("<error N>" for a rejected word).
std::vector<std::string> analyze_compact(crispembed_ar_morph * m, const std::vector<std::string> & words,
                                         uint32_t flags = 0) {
    std::string concat;
    std::vector<uint32_t> lens;
    for (const std::string & w : words) {
        concat += w;
        lens.push_back(uint32_t(w.size()));
    }
    unsigned char * buf = nullptr;
    size_t len = 0;
    char err[256];
    std::vector<std::string> out;
    const int rc = crispembed_ar_morph_analyze_compact(m, concat.data(), lens.data(), uint32_t(lens.size()), flags,
                                                       &buf, &len, err, sizeof err);
    if (rc != CRISPEMBED_AR_MORPH_OK) return out;
    const std::vector<std::string> keys = info_keys(crispembed_ar_morph_info(m));
    const unsigned char * p = buf;
    const unsigned char * end = buf + len;
    if (rd32(p) != CRISPEMBED_AR_MORPH_COMPACT_MAGIC || rd32(p + 4) != words.size()) return out;
    p += 8;
    for (size_t w = 0; w < words.size(); ++w) {
        const uint32_t status = rd32(p), bytes = rd32(p + 4);
        p += 8;
        if (status != CRISPEMBED_AR_MORPH_OK) {
            out.push_back("<error " + std::to_string(status) + ">");
            p += bytes;
            continue;
        }
        const unsigned char * rec = p;
        const uint32_t n = rd32(rec), n_strings = rd32(rec + 4);
        const bool wide = rec[8] != 0;
        rec += 12;
        std::vector<std::string> strs;
        for (uint32_t i = 0; i < n_strings; ++i) {
            const uint32_t l = rd32(rec + 1);
            const std::string text(reinterpret_cast<const char *>(rec + 5), l);
            strs.push_back(rec[0] == 1 ? text : "\"" + core_json::json_escape_strict(text) + "\"");
            rec += 5 + l;
        }
        std::string json = "[";
        for (uint32_t a = 0; a < n; ++a) {
            json += a ? ",{" : "{";
            const uint32_t nf = rd16(rec);
            rec += 2;
            for (uint32_t f = 0; f < nf; ++f) {
                const uint32_t key = wide ? rd16(rec) : rec[0];
                uint32_t ref = wide ? rd32(rec + 2) : rd16(rec + 1);
                if (!wide && ref == 0xFFFF) ref = 0xFFFFFFFFu;
                rec += wide ? 6 : 3;
                if (f) json += ',';
                json += "\"" + core_json::json_escape_strict(keys.at(key)) + "\":";
                json += ref == 0xFFFFFFFFu ? std::string("null") : strs.at(ref);
            }
            json += '}';
        }
        json += ']';
        out.push_back(rec == p + bytes ? json : std::string("<bad record>"));
        p += bytes;
    }
    if (p != end) out.clear();
    crispembed_ar_morph_buffer_free(buf);
    return out;
}

// Compact output equals the JSON ABI on every fixture word plus rejects.
void check_compact(crispembed_ar_morph * m, int backoff, const char * what) {
    std::vector<std::string> words;
    for (const expected_case & c : k_cases)
        if (c.backoff == backoff) words.push_back(c.word);
    words.push_back(std::string("\xff\xfe"));
    words.push_back(std::string());
    std::vector<std::string> want;
    for (const std::string & w : words) want.push_back(analyze(m, w));
    check(analyze_compact(m, words) == want, what);
    check(analyze_compact(m, words, CRISPEMBED_AR_MORPH_COMPACT_DEDUP) == want, "dedup keeps distinct analyses");
}

void check_binary(const std::string & manifest_dir) {
    const std::string bin(reinterpret_cast<const char *>(k_bin_db), sizeof k_bin_db);
    const std::string path = temp_path("crispembed-ar-morph-test.cmorph");
    check(write_file(path, bin), "write binary fixture");
    const std::string sha = core_sha256::bytes(bin.data(), bin.size());
    char err[256];
    crispembed_ar_morph * none = crispembed_ar_morph_open_db(path.c_str(), sha.c_str(), 0, err, sizeof err);
    crispembed_ar_morph * prop = crispembed_ar_morph_open_db(path.c_str(), sha.c_str(), 2, err, sizeof err);
    check(none && prop, "open binary fixture");
    if (!none || !prop) {
        std::fprintf(stderr, "binary open: %s\n", err);
        return;
    }
    const std::string info = crispembed_ar_morph_info(none);
    check(info.find("\"db_format\":\"binary\"") != std::string::npos, "binary info format");
    check(info.find("\"stem_entries\":3") != std::string::npos, "binary info counts");
    for (const bin_case & c : k_bin_cases) {
        const std::string got = analyze(c.backoff == 0 ? none : prop, c.word);
        if (got != c.json) {
            std::fprintf(stderr, "binary word %s\n got: %s\nwant: %s\n", c.word, got.c_str(), c.json);
            check(false, "binary camel_tools parity (profile)");
        }
    }
    int rc = 0;
    analyze(prop, std::string("\xd8\xb2\\q"), &rc);
    check(rc == CRISPEMBED_AR_MORPH_INPUT_ERROR, "binary: invalid template raises like camel_tools");
    check_compact(none, 0, "binary compact equals JSON");
    check_compact(prop, 2, "binary compact equals JSON (backoff)");
    crispembed_ar_morph_free(prop);

    // The heap path (hosts without mmap) gives the same results.
    set_env("CRISPEMBED_AR_MORPH_MMAP", "0");
    crispembed_ar_morph * heap = crispembed_ar_morph_open_db(path.c_str(), sha.c_str(), 0, err, sizeof err);
    set_env("CRISPEMBED_AR_MORPH_MMAP", nullptr);
    check(heap != nullptr, "open binary fixture on the heap");
    if (heap) {
        check(std::string(crispembed_ar_morph_info(heap)).find("\"storage\":\"heap\"") != std::string::npos,
              "heap storage reported");
        check(analyze(heap, k_bin_cases[0].word) == k_bin_cases[0].json, "heap analysis");
        crispembed_ar_morph_free(heap);
    }
    crispembed_ar_morph_free(none);

    // Corruption: every section and the header are checksummed and bounded.
    for (size_t at : { size_t(3), size_t(30), size_t(100), bin.size() / 2, bin.size() - 5 }) {
        std::string bad = bin;
        bad[at] = char(bad[at] ^ 0x5a);
        check(write_file(path, bad), "write corrupt binary");
        const std::string bsha = core_sha256::bytes(bad.data(), bad.size());
        check(!crispembed_ar_morph_open_db(path.c_str(), bsha.c_str(), 0, err, sizeof err), "corrupt binary rejected");
    }
    {
        const std::string cut = bin.substr(0, bin.size() - 8);
        check(write_file(path, cut), "write truncated binary");
        const std::string csha = core_sha256::bytes(cut.data(), cut.size());
        check(!crispembed_ar_morph_open_db(path.c_str(), csha.c_str(), 0, err, sizeof err),
              "truncated binary rejected");
    }
    check(write_file(path, bin), "rewrite binary fixture");

    // A version-2 bundle opens the binary database through its manifest.
    const std::string name = "crispembed-ar-morph-test.cmorph";
    auto manifest_for = [&](const char * format) {
        return std::string("{\"format\":\"crispembed-camel-morph-bundle\",\"format_version\":2,") +
               "\"variant\":\"proofing-test\",\"db\":\"" + name + "\"" + format + ",\"files\":[{\"path\":\"" + name +
               "\",\"size\":" + std::to_string(bin.size()) + ",\"sha256\":\"" + sha + "\"}]}";
    };
    const std::string mpath = manifest_dir + "crispembed-ar-morph-bin-manifest.json";
    const std::string manifest = manifest_for(",\"db_format\":\"crispembed-camel-morph-bin-v1\"");
    check(write_file(mpath, manifest), "write binary manifest");
    const std::string msha = core_sha256::bytes(manifest.data(), manifest.size());
    crispembed_ar_morph * bundled = crispembed_ar_morph_open(mpath.c_str(), msha.c_str(), 0, err, sizeof err);
    check(bundled != nullptr, "open binary bundle");
    if (bundled) {
        check(analyze(bundled, k_bin_cases[0].word) == k_bin_cases[0].json, "binary bundle analysis");
        crispembed_ar_morph_free(bundled);
    }
    const std::string untyped = manifest_for("");
    check(write_file(mpath, untyped), "write untyped manifest");
    const std::string usha = core_sha256::bytes(untyped.data(), untyped.size());
    check(!crispembed_ar_morph_open(mpath.c_str(), usha.c_str(), 0, err, sizeof err), "db_format required");
    std::remove(mpath.c_str());
    std::remove(path.c_str());
}

} // namespace

static int crispembed_test_main() {
    // The raw literal starts with a newline after R"DB(.
    const std::string db = std::string(k_db).substr(1);
    const std::string db_path = temp_path("crispembed-ar-morph-test.db");
    check(write_file(db_path, db), "write fixture");
    const std::string sha = core_sha256::bytes(db.data(), db.size());
    char err[256];

    check(crispembed_ar_morph_abi_version() == CRISPEMBED_AR_MORPH_ABI_VERSION, "abi version");
    const std::string wrong(64, '0');
    check(!crispembed_ar_morph_open_db(db_path.c_str(), wrong.c_str(), 0, err, sizeof err), "wrong hash rejected");
    check(std::strstr(err, "SHA-256") != nullptr, "hash error message");
    check(!crispembed_ar_morph_open_db(db_path.c_str(), "ABC", 0, err, sizeof err), "malformed hash rejected");
    check(!crispembed_ar_morph_open_db(db_path.c_str(), sha.c_str(), 9, err, sizeof err), "bad backoff rejected");

    crispembed_ar_morph * none = crispembed_ar_morph_open_db(db_path.c_str(), sha.c_str(), 0, err, sizeof err);
    crispembed_ar_morph * prop = crispembed_ar_morph_open_db(db_path.c_str(), sha.c_str(), 2, err, sizeof err);
    check(none && prop, "open fixture");
    if (!none || !prop) return 1;
    check(std::strstr(crispembed_ar_morph_info(none), "\"stem_entries\":3") != nullptr, "info counts");

    for (const expected_case & c : k_cases) {
        const std::string got = analyze(c.backoff == 0 ? none : prop, c.word);
        if (got != c.json) {
            std::fprintf(stderr, "word %s\n got: %s\nwant: %s\n", c.word, got.c_str(), c.json);
            check(false, "camel_tools parity");
        }
    }

    int rc = 0;
    analyze(none, std::string("\xff\xfe"), &rc);
    check(rc == CRISPEMBED_AR_MORPH_INPUT_ERROR, "invalid UTF-8 rejected");
    analyze(none, std::string(CRISPEMBED_AR_MORPH_MAX_WORD_BYTES + 1, 'a'), &rc);
    check(rc == CRISPEMBED_AR_MORPH_INPUT_ERROR, "long word rejected");
    check(analyze(none, std::string()) == "[]", "empty word");
    // A backoff stem that is an invalid re.sub template makes camel_tools raise.
    analyze(prop, std::string("\xd8\xb2\\q"), &rc);
    check(rc == CRISPEMBED_AR_MORPH_INPUT_ERROR, "invalid template raises like camel_tools");

    // Bundle manifest: authenticated manifest, per-file size and hash.
    const std::string notice = "notice\n";
    const std::string notice_path = temp_path("NOTICE");
    check(write_file(notice_path, notice), "write notice");
    const std::string db_name = "crispembed-ar-morph-test.db";
    auto manifest_for = [&](const std::string & db_sha) {
        return std::string("{\"format\":\"crispembed-camel-morph-bundle\",\"format_version\":1,\"variant\":\"test\",") +
               "\"db\":\"" + db_name + "\",\"files\":[{\"path\":\"NOTICE\",\"size\":" + std::to_string(notice.size()) +
               ",\"sha256\":\"" + core_sha256::bytes(notice.data(), notice.size()) + "\"},{\"path\":\"" + db_name +
               "\",\"size\":" + std::to_string(db.size()) + ",\"sha256\":\"" + db_sha + "\"}]}";
    };
    const std::string manifest = manifest_for(sha);
    const std::string manifest_path = temp_path("crispembed-ar-morph-manifest.json");
    check(write_file(manifest_path, manifest), "write manifest");
    const std::string msha = core_sha256::bytes(manifest.data(), manifest.size());
    crispembed_ar_morph * bundled = crispembed_ar_morph_open(manifest_path.c_str(), msha.c_str(), 0, err, sizeof err);
    check(bundled != nullptr, "open bundle");
    if (bundled) {
        check(analyze(bundled, "\xd9\x88\xd9\x83\xd8\xaa\xd8\xa7\xd8\xa8") == k_cases[0].json, "bundle analysis");
        crispembed_ar_morph_free(bundled);
    }
    check(!crispembed_ar_morph_open(manifest_path.c_str(), wrong.c_str(), 0, err, sizeof err), "manifest hash");
    const std::string tampered = manifest_for(wrong);
    check(write_file(manifest_path, tampered), "write tampered manifest");
    const std::string tsha = core_sha256::bytes(tampered.data(), tampered.size());
    check(!crispembed_ar_morph_open(manifest_path.c_str(), tsha.c_str(), 0, err, sizeof err), "db pin enforced");

    check_compact(none, 0, "text compact equals JSON");
    check_compact(prop, 2, "text compact equals JSON (backoff)");
    crispembed_ar_morph_free(none);
    crispembed_ar_morph_free(prop);
    check_binary(temp_path(""));
    std::remove(db_path.c_str());
    std::remove(notice_path.c_str());
    std::remove(manifest_path.c_str());
    if (failures) return 1;
    std::printf("arabic-morph contract: OK\n");
    return 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
