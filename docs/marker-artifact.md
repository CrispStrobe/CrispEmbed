# Native Marker extraction artifacts

An extraction artifact stores the native Marker document and extraction progress
in one self-contained file. It can retain completed pages when a longer job is
cancelled or interrupted, and it can reopen a processed result for another
Markdown, HTML, JSON or chunk rendering without OCR inference. The suggested
extension is `.cemk`; readers inspect the contents, not the extension.

The format is an original native implementation. PDF Craft's separation of
extraction from later rendering informed the design; this is not its PCEX format
and does not import its Python, Qt, Poppler or model dependencies. See the
[source and license review](pdf-craft-adoption.md). The implementation uses the
repository's existing streaming SHA-256 helper and introduces no dependency.

## API and access

The internal C++ contract in `src/marker_artifact.h` is:

```cpp
namespace crisp_marker {
struct extraction_snapshot {
    document value;
    std::map<std::string, std::string> identity;
    int first_page = 0, last_page = -1, next_page = 0;
    bool complete = false;
};

void save_extraction(const std::string & path,
                     const extraction_snapshot & snapshot,
                     const artifact_limits & limits = {});
extraction_snapshot load_extraction(const std::string & path,
                                   const artifact_limits & limits = {});
std::string extraction_file_sha256(const std::string & path);
}
```

These helpers perform no model initialization and do not inspect environment
gates. The public C interface requires `CRISPEMBED_MARKER=1` and
`CRISPEMBED_MARKER_ARTIFACTS=1`. Incremental extraction additionally requires
`CRISPEMBED_MARKER_INCREMENTAL=1`; OCR inference keeps its separate model feature
and research-license acknowledgement requirements. Reopening already extracted
content in a text-mode context does not load a model or require model
acknowledgement. This opens the extracted document explicitly selected by the
caller; it does not grant access to restricted model weights.

`load_extraction` returns a fully checked snapshot or throws. It never returns a
partially decoded result. `save_extraction` validates before creating its
temporary file and throws on encoding, resource-limit or I/O failure. Callers
must not mutate a snapshot concurrently with saving it.

## What is preserved

Version 1 serializes every field of the native `document` IR:

- Document source name, backend, metadata, warnings and immutable source records.
- Page indices, dimensions, PDF text lines and spans, extraction method, original
  layout bytes, layout token counts and cap warnings, printed labels and their
  evidence/candidates.
- The recursive block tree, identifiers, type and order, current text, raw OCR
  bytes, raw labels, HTML field, source, rectangles, polygons, lines, span fonts
  and URLs, section hierarchy, image names, scores, tokens, flags and attributes.
- Source ranges, continuation targets, footnote identities and reference ranges,
  resolution status, target and candidate identities.
- Embedded asset names, MIME types and complete original byte payloads.

Text and metadata are length-prefixed bytes. Embedded NUL bytes and invalid UTF-8
round-trip unchanged; the artifact layer does not normalize or repair them.
Source ranges use half-open byte offsets against the current block text and the
referenced immutable source text. `copy` ranges must match those bytes exactly.
`whitespace` ranges may map ASCII space, tab, CR and LF to one space or to an empty
output range. Present ranges must cover current text contiguously. Missing source
IDs, invalid offsets, unknown persisted transforms and non-finite numbers are
rejected. Older unreferenced source records remain valid history. Geometry belongs
to the complete recorded source block; a byte slice does not create a new box.

The artifact stores content, not an executable object graph. Asset names are
unique flat member names: slash, backslash, colon, control characters, `.` and
`..` are rejected. Loading retains assets in memory and does not create files
from their names.

## Progress, phases and identity

Page indices are physical, zero-based indices. `first_page` and `last_page` are
inclusive; `next_page` is the first page not yet committed. Stored pages must be
contiguous, start at `first_page`, and number exactly
`next_page - first_page`. An initial incomplete checkpoint may contain no pages.
An incomplete internal snapshot may use `last_page=-1` for an unknown endpoint;
public incremental jobs record the known endpoint. A complete snapshot has a
known nonempty selected range and `next_page == last_page + 1`.

`complete` says extraction reached the selected endpoint. Rendering state is a
separate `metadata["extraction_phase"]` value:

| Phase | Stored document | Public behavior |
| --- | --- | --- |
| `raw` | Completed extraction units before document-wide processing | Resume uses this checkpoint. Opening a view requires the same processing fingerprint and processes a copy. |
| `processed` | Current document after deterministic structure/source processing | Reopens directly for rendering and retains its source maps. It cannot be used to resume extraction. |

Public incremental extraction commits each completed page before reporting that
page as durable progress. A processed view is saved separately from its raw
checkpoint. A complete raw checkpoint remains resumable/openable; completion
does not silently change its phase. Internal serialization treats metadata as
content; the public lifecycle validates these phase values.

The public lifecycle records source SHA-256, model SHA-256 or `none`, an explicitly
selected PDFium library hash, the processing fingerprint, the native build
identity and the extraction contract. Resumption compares identities and page
selection before continuing. The processing fingerprint includes configuration
and relevant feature gates; password and absolute source/model/library paths
are excluded. The format contains no serialized `options` object. Internal
callers remain responsible for not placing credentials in free-form document
metadata or identity fields.

The file-hash helper streams arbitrary source/model/library files in 64 KiB
blocks and follows explicitly selected symlinks to their regular-file targets.
Artifact input and output paths themselves reject final-component symlinks and
nonregular files. An opened processed artifact needs neither the original source
nor the original model file. Its identity is retained provenance; reopening
alone does not independently authenticate the original document or its author.

## Portable version 1 encoding

The fixed header is 88 bytes:

| Offset | Bytes | Meaning |
| --- | ---: | --- |
| 0 | 8 | `CRISPMK` followed by a zero byte |
| 8 | 4 | Little-endian unsigned version, currently `1` |
| 12 | 4 | Little-endian unsigned flags, currently `0` |
| 16 | 8 | Little-endian unsigned payload length |
| 24 | 64 | Lowercase ASCII hexadecimal SHA-256 of header bytes 0–23 followed by the payload |
| 88 | Declared length | Snapshot payload |

The payload begins with `first_page`, `last_page`, `next_page`, `complete`, the
identity map, and the document. Integer fields use signed 32-bit little-endian
encoding. Counts, string/blob lengths and byte offsets use unsigned 64-bit
little-endian encoding. Booleans use one byte, exactly `0` or `1`. Finite doubles
use their IEEE-754 binary64 bits, stored little-endian; negative zero is retained.
Vectors preserve order. Maps are encoded in native sorted-key order and reject
duplicate keys while decoding. Field order is explicit in
`transfer_snapshot`, `transfer_document`, `transfer_block` and `transfer_line`
in `src/marker_artifact.cpp`; compiler structure padding is never serialized.

Readers reject unknown versions/flags, truncated fields, undeclared trailing
bytes, leftover payload fields and integrity mismatches. There is no version
migration yet. SHA-256 detects accidental changes and incomplete writes; it is
an integrity digest, not a signature or authentication mechanism. Structural
and resource checks apply even when a file's author recomputes its digest.

## Resource limits and filesystem commit

The default `artifact_limits` are:

| Limit | Default |
| --- | ---: |
| Entire encoded file | 256 MiB |
| One string | 4 MiB |
| All embedded asset bytes | 128 MiB |
| Pages | 1,000 |
| Blocks per page, including descendants | 4,096 |
| Aggregate vector/map entries | 1,000,000 |
| Block nesting depth | 32 |

Identity maps additionally allow at most 64 entries, keys of at most 128 ASCII
identifier bytes (`A-Z`, `a-z`, `0-9`, `_`, `-`, `.`), and values of at most 4,096
printable ASCII bytes. Identity lengths are checked before allocation. All
collection lengths, strings and blobs are bounded against remaining payload
bytes before their data is allocated; collection members decode incrementally.
Source-map byte comparisons share a work budget equal to the configured file
limit, preventing repeated zero-output ranges from causing unbounded scanning.
Configuration rejects a file cap above 4 GiB or nesting above 64. Public callers
can impose narrower page, block and asset caps before decoding. These storage
limits supplement the separate processing/source-link limits.

The file cap measures encoded bytes, not peak RAM. Incremental extraction retains
a raw document and a processed view; reopening and rendering also allocate IR,
metadata and output strings. Each committed page rewrites the complete snapshot.
The page budget bounds extraction units per call, while total memory and save
time still depend on the retained document within these limits.

Saving first validates and measures the complete encoding. It then exclusively
creates a sibling temporary file in the destination directory, writes and hashes
the payload, flushes and syncs the file, closes it, and replaces the destination.
POSIX uses `rename`; Windows uses `MoveFileExW` with replacement and write-through
flags. A pre-commit failure removes the owned temporary file and preserves any
previous destination. The POSIX parent directory is synced on a best-effort basis
after commit. Power-loss durability still depends on the filesystem and storage
device; the API does not claim a cross-filesystem transaction. Callers should use
a directory they control and coordinate competing writers.

## Validation status

`tests/test-marker-artifact.cpp` contains owned, model-free fixtures for every IR
field, arbitrary bytes, incomplete/complete and raw/processed phases, an initial
empty checkpoint, source maps including removed whitespace and retained history,
deterministic encoding, the standard SHA-256 `abc` vector, and atomic replacement.
Malformed fixtures include invalid versions/flags, digest corruption, truncation,
trailing data, malicious counts/identity lengths, duplicate keys, forged source
maps, unsafe member paths, inconsistent page progress, tighter resource limits,
non-finite geometry and rejected archive symlinks. Failed-save fixtures verify
that previous bytes remain unchanged and temporary siblings are absent.

The owner-authorized Linux CPU run passes all 87 artifact checks at native source
`d4d508c460c6e172fd1d7d4afc6b9c78652ad607`, together with 400 native lifecycle
checks and actual Python/Rust/Dart reopen controls. The first artifact and
lifecycle attempts failed before any checks because the environment had no
default temporary directory. Both pass unchanged with a task-local `TMPDIR`;
the [execution receipt](../tests/results/marker/pdf-craft-local-2026-11-09/README.md)
retains both attempts. Windows-specific filesystem execution remains pending.
