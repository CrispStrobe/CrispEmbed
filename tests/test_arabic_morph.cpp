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
#include "arabic_morph.h"
#include "core/sha256.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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

} // namespace

int main() {
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

    crispembed_ar_morph_free(none);
    crispembed_ar_morph_free(prop);
    std::remove(db_path.c_str());
    std::remove(notice_path.c_str());
    std::remove(manifest_path.c_str());
    if (failures) return 1;
    std::printf("arabic-morph contract: OK\n");
    return 0;
}
