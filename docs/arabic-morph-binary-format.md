# Camel Morph binary database format (`crispembed-camel-morph-bin-v1`)

The native Camel Morph analyzer (`src/arabic_morph.cpp`) reads two database
representations behind the same C ABI:

- the ALMOR **text** database (camel_tools' format), parsed into heap tables,
  opened from a `format_version: 1` bundle or a bare `.db` file;
- this **binary** format, memory-mapped and used in place, opened from a
  `format_version: 2` bundle or a bare file that starts with the magic.

`models/convert-camel-morph-db.py` writes the binary file from the pinned
`noprob` text database. Its parser mirrors the native text loader, which is
itself checked against camel_tools. The output is byte-identical on every run:
strings are sorted, rows are grouped by form, and there are no timestamps.

## What the file holds

The file holds the analyzer's `MorphologyDB` reduced to a **feature profile**:

- prefixes, suffixes and stems with their categories;
- the AB, BC and AC category-compatibility tables;
- the DEFAULT entries (restricted to the profile) and the STEMBACKOFF lists;
- for every entry, only the profile features, plus `bw` on the `NOAN` backoff
  stems.

The analyzer runs camel_tools' algorithm unchanged, but skips every
assignment of a feature outside the profile. No profile value is computed
from a dropped feature, so analyses and their profile values are exactly
camel_tools'. The converter enforces the two dependencies that exist:

- `gen` needs `form_gen`, and `num` needs `form_num` (`fix_form`);
- the backoff modes test `NOUN_PROP in bw` on the NOAN stems, so `bw` is
  stored on those rows only and never output.

The proofing profile used for Quds Office is `diac lex root pos prc3 prc2
prc1 prc1.5 prc0 per asp vox mod form_gen gen form_num num stt cas enc0 enc1
enc2 rat source`.

Output key order is each stem's own key order. The converter puts the columns
in an order consistent with every stem (ties follow the database's ORDER
line) and refuses inconsistent databases. An analysis therefore lists its
profile features in camel_tools' order.

## Layout

All integers are little-endian. Sections start at 8-byte-aligned offsets.
A big-endian host refuses the file.

### Header (64 bytes)

| Offset | Type | Field |
| ---: | --- | --- |
| 0 | `char[8]` | magic `CEMORPHB` |
| 8 | u32 | format version, `1` |
| 12 | u32 | header size, `64` |
| 16 | u32 | section count (at most 64) |
| 20 | u32 | flags: bit 0 = `pattern` is in ORDER and in the profile |
| 24 | u64 | file size |
| 32 | u32 | CRC-32 (IEEE) of the section directory |
| 36 | u32 | CRC-32 of the 64 header bytes with this field zeroed |
| 40 | `u8[24]` | zero |

### Section directory

The directory follows the header and has one 24-byte entry per section:
`u32 tag, u32 CRC-32 of the section, u64 offset, u64 size`. Tags are four
ASCII bytes read as a little-endian u32. Every tag below must occur exactly
once.

| Tag | Content |
| --- | --- |
| `STRS` | `u32 n`, `u32 offsets[n+1]`, then the UTF-8 bytes. Strings are unique and sorted by bytes; `offsets[0] = 0`, offsets never decrease, and string *i* is `bytes[offsets[i]..offsets[i+1])`. Every value, form, category, key and backoff name is a string id. |
| `KEYS` | `u32 n`, then `n × {u32 name id, u32 flags}`. Flag 1 means DEFINEd in the database, flag 2 means output (in the profile). Entry *i* is "key index *i*". |
| `CATS` | `u32 n`, `u32 name id[n]`: category names; entry *i* is category id *i*. |
| `DFLT` | `u32 n`, then per DEFAULT: `u32 pos value id, u32 m, m × {u32 key index, u32 value id or 0xFFFFFFFF for None}` (output keys only, in the line's order). |
| `BKOF` | `u32 n`, then per STEMBACKOFF: `u32 name id, u32 m, u32 category-name id[m]`. |
| `PREF`, `SUFF`, `STEM` | Entry tables (below). |
| `TBAB`, `TBBC`, `TBAC` | `u32 n, u32 0`, then `n` u64 pairs `(a << 32) \| b` of category ids, strictly ascending. |

### Entry tables

```
u32 rows, u32 forms, u32 columns, u32 row_bytes
columns × { u32 key index (0xFFFFFFFF = category column), u32 width (1|2|4),
            u32 offset in the row, u32 dict_count }
the column dictionaries, in column order: dict_count × u32
    (string ids; category ids for the category column)
pad to 8
forms × { u32 form string id, u32 first row, u32 row count }
pad to 8
rows × row_bytes
```

- A row stores, for each column, a local index into that column's
  dictionary, in the smallest width that holds it.
- The all-ones value of the width (`0xFF`, `0xFFFF`, `0xFFFFFFFF`) means the
  entry lacks the feature. This is distinct from an empty string, which is a
  dictionary value.
- Exactly one column is the category column, and it is never absent.
- Forms are sorted by string id, which is byte order. Each form's rows are
  contiguous, in database file order, and together they cover all rows.
- Lookups binary-search the form index.
- Entry ids are global: prefixes first, then suffixes, then stems.

## Validation at load

The loader checks the following before any lookup, so a truncated or corrupt
file is rejected, never read out of bounds:

- magic, version, header size and file size;
- header and directory CRCs, then every section CRC;
- section bounds and alignment;
- string offsets, UTF-8 validity and sort order;
- every key, string, category and column-dictionary id;
- column layout and the form index (ordering, contiguity, coverage);
- every row value, which must be a valid local index or absent;
- that the compatibility pairs are ascending.

A bundle additionally pins the whole file by SHA-256 in `manifest.json`,
whose own SHA-256 the host supplies. For the 12 MB proofing database,
validation and hashing take about 0.1 s on an Apple M1.

## Storage

On POSIX hosts the file is mapped with `mmap(PROT_READ, MAP_PRIVATE)`, and on
Windows with `MapViewOfFile`. No table is copied to the heap; the small
key, default and backoff maps are the only heap structures.

`CRISPEMBED_AR_MORPH_MMAP=0`, and Emscripten builds, read the file into one
heap buffer instead. `crispembed_ar_morph_info` reports which path was used
as `"storage": "mapped" | "heap"`.

## Bundle manifest (`format_version: 2`)

The bundle manifest has the same shape as version 1, with these additions:

- `"db_format": "crispembed-camel-morph-bin-v1"`;
- `"profile"`: the output features;
- `"derived_from"`: the noprob database it was converted from, with size and
  SHA-256.

The loader accepts version 1 (text) and version 2 (binary) manifests.

## Compact output

`crispembed_ar_morph_analyze_compact` (ABI version 2) analyzes a batch of
words and returns one buffer. The analyses and values are the same as
`crispembed_ar_morph_analyze`'s JSON, with no JSON involved. The layout is
documented in `src/arabic_morph.h`:

- each word has a record of unique value strings, followed by its analyses as
  (key id, string index) pairs;
- key ids index the `keys` array of `crispembed_ar_morph_info`;
- `CRISPEMBED_AR_MORPH_COMPACT_DEDUP` drops repeated analyses of a word.
  After projection onto the proofing profile, that is 6% of them on UD
  Arabic-PUD.
