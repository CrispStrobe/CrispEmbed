# Restored W7 package and complete-byte rebuilds — 18 November 2026

The current saved packager passes all 25 model-free controls. An isolated CPython 3.12.14 environment installed the unchanged hash-locked requirements using only wheels. The official model wheel matches its fixed 14,639,490-byte SHA-256. The tracked converter exported all 39 F32 tensors and independently read them back; all five resulting bundle files match the retained identities exactly.

The accepted native archive was authenticated before parsing and restored without rebuilding. All 15 native files match. Readelf and the authenticated library's loader-resolution inspection establish the same ELF requirements, no RPATH/RUNPATH, and all six exact external dependency identities on this Linux x86-64 host. The candidate native library was not loaded and no inference was run by this restoration.

Two new complete packages were independently built to separate new filenames from the actual native/model inputs. Their complete bytes are equal. Each is 7,808,129 bytes with SHA-256 `4e3b627a614cd21851e6e354335e2e08adeb4cedd4d7778df0cc68c1ce2edf01`, matching the retained original package identity. The earlier workspace's archive was unavailable; comparison against that original uses its saved full SHA-256 and length, while the two new archives were additionally compared byte for byte.

Atomic installation and full installed-tree verification pass with the existing fixed payload pins. The installer uses the host's no-replace directory publication primitive; it does not load the spaCy candidate. Generated model weights, full archives and the isolated reference environment remain outside ordinary Git.

`package-result.json` binds the exact converter, lock, packager and control-source hashes. `evidence-index.json` hashes the retained evidence. Receipts replace local runtime paths with `<runtime-spacy>` and repository paths with `<CrispEmbed>`; these are placeholders rather than an executable machine-specific dependency. Dependency filenames and original byte identities remain explicit.

This completes the pending full-input package rebuild comparison. It does not establish operational-consumer execution, broaden Linux/device portability, rerun either consumed linguistic panel, publish a release or enable a default. No GitHub Actions endpoint was queried, awaited, rerun or dispatched.
