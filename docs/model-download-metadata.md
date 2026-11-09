# Read-only model download metadata

The feature branch adds an explicit, metadata-only API for applications that own
their model installation directory and network client. It exposes the existing
registry's exact filename, source URL, SHA-256 pin, license information and any
declared companion. The call does not resolve a cache, download a file, prompt
for or record license acceptance, change an environment variable or load a model.

## Rust interface

```rust
let info = crispembed::CrispEmbed::model_download_info("ppocrv6-tiny-rec")?
    .ok_or("Unknown registered model")?;
assert_eq!(info.name, "ppocrv6-tiny-rec");

for file in std::iter::once(&info.primary).chain(info.companions.iter()) {
    let expected_sha256 = file.sha256.as_deref().ok_or("Model is not pinned")?;
    // Validate the filename and HTTPS URL, stream to an app-private temporary
    // file, verify expected_sha256, then atomically install the complete set.
}
```

`ModelDownloadInfo` contains `name`, `license`, `model_card_url`,
`requires_license_acceptance`, `primary: ModelDownloadFile`, and
`companions: Vec<ModelDownloadFile>`. Each file contains `filename`, `url`
and `sha256: Option<String>`. These are owned copies, safe to retain across
other metadata calls and to read independently on different threads.

The free function `crispembed::model_download::model_download_info` has the
same behavior. A lookup uses the exact, case-sensitive registry name. It returns
`Ok(None)` for an unknown name, filename, substring or path; an embedded NUL
byte returns an error. It intentionally does not inherit fuzzy model resolution.

## C interface and lifetime

`crispembed_get_model_download_info_v1(exact_name, &out)` returns `1` for
an exact match, `0` for an unknown name and `-1` for a NULL argument. A
non-NULL output is cleared before lookup. The new fixed v1 structs are additive;
future layout extensions require a new version instead of changing this ABI.

All non-NULL strings borrow immutable registry storage valid for process
lifetime. A NULL SHA means the registry has no pin. A companion declaration is
surfaced even if one of its strings is missing; the installer must reject an
incomplete payload set instead of silently dropping a required file. The current
registry has at most one companion per entry; Rust exposes a vector for the
application's installation loop.

## Installer responsibilities

A registry entry is metadata, not approval to download or execute a model.
Honor `requires_license_acceptance` and the actual license/model card before
installation. Reject unsafe filenames, non-HTTPS sources, missing or invalid
pins and redirects that violate the application's transport policy. Stream
large payloads with cancellation and bounded memory; verify their SHA-256
against the actual registry pin before an atomic install. Install all required
companions alongside the primary file and pass the resulting absolute primary
path to the native loader.

The API returns existing pins unchanged; it never hashes a URL, trusts an ETag
as a new digest, synthesizes a pin or treats an unpinned payload as verified.
It also does not change the legacy downloader's behavior. This lets mobile
applications use their own HTTP/cache implementation without native subprocesses
or a process-global cache-directory race.

## Qualification boundary

The [hosted metadata contract](../.github/workflows/model-download-metadata.yml)
builds the production model manager and C declaration, checks every registered
entry and its available pins, verifies exact-name rejection and a real companion,
runs ASan/UBSan, and links the actual safe Rust wrapper for ownership, absent-pin
translation and simultaneous lookup controls. It does not download weights.
Final-source hosted results are required before declaring this API validated.

App download cancellation, storage limits, hash rejection, atomic recovery,
same-source mobile linking, device OCR execution and model quality remain
separate consumer gates. No release, model alias or inference default follows
from this additive metadata API.
