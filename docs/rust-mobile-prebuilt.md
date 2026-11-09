# Rust mobile native libraries

The feature branch adds an explicit prebuilt contract to `crispembed-sys`.
Cargo's `TARGET`, rather than the host that runs its build script, selects the
native library format and additional system libraries. This is source work on
the feature branch; no published crate, app release or engine default is changed.

| Cargo target | Required file in `CRISPEMBED_SYS_LIB_DIR` | Native link contract |
| --- | --- | --- |
| iOS device or simulator | `libcrispembed.a` | A complete merged archive containing CrispEmbed, core and required ggml objects; link C++, Foundation, Metal, MetalKit and Accelerate. |
| Android | `libcrispembed.so` | A target shared library with verified dependencies, including static ggml and static libc++; the app packages the shared library for the selected Android ABI. |
| Other cross compilation | Target shared library or Windows import library | The explicitly supplied target closure; no automatic host build. |
| Native desktop | Existing shared library discovery | Existing explicit-directory, in-tree and source-build fallback behavior. |

For example, after building and verifying the Android target closure:

```sh
CRISPEMBED_SYS_LIB_DIR="$PWD/mobile-native/lib" \
  cargo build --target aarch64-linux-android
```

The iOS archive must already contain all required static dependencies. Supplying
only `libcrispembed-static.a`, renaming a partial archive, or using a macOS library
does not satisfy this contract. The target linker resolves the archive against
the platform SDK. Framework directives do not turn on an engine or change its
runtime backend selection; they satisfy the symbols of a complete native archive.

For Android, this contract assumes the C++ runtime is included in the verified
shared-library closure. It does not add host `stdc++` or an unbundled
`libc++_shared.so`. A producer that uses a different closure must supply and
validate that complete packaging configuration before adopting it.

Mobile and other cross builds require `CRISPEMBED_SYS_LIB_DIR` even if a host
`build/` directory exists. A missing, empty or incompatible directory fails
before directory discovery or CMake. This prevents accidental host libraries and
unconfigured host CMake from entering a mobile build. Library filename presence
is a configuration check; source identity, architecture, archive completeness,
Android dependency closure and actual target linking remain producer/consumer
qualification gates.

## Evidence and remaining gates

The [target prebuilt workflow](../.github/workflows/rust-prebuilt-contract.yml)
exercises the production link policy and the actual build script without model
inference. It covers device/simulator plans, all supported Android ABI plans,
target-specific desktop cross libraries, required explicit paths, unchanged
desktop fallback, emitted link directives and missing/wrong-library rejection.
Its empty placeholder files are configuration controls, never native binaries.

Actual mobile cross linking must consume the same immutable source revision and
verified native producer output. App packaging, model provisioning, startup,
camera/files access and OCR execution on devices are separate gates. No mobile
execution result follows from the configuration controls or a successful link.
