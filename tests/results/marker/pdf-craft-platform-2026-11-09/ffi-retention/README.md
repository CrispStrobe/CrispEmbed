# Apple FFI retention controls

The CocoaPods follow-up found two separate static-link requirements. The final
application must load the native archive even though Dart names functions only
at runtime; the loaded public functions must also survive Release dead
stripping. The original iOS podspec placed its force-load flag in the private pod
target configuration. CocoaPods' [aggregate build settings implementation](https://github.com/CocoaPods/CocoaPods/blob/b80e113e28a7e23cd6dbc169f0c923f6b48bf7a2/lib/cocoapods/target/build_settings.rb)
propagates `user_target_xcconfig` to the application. `PODS_TARGET_SRCROOT` is a
pod-target variable, so simply moving the original flag would not resolve its
archive in the application.

The corrected podspec computes the archive's path relative to the actual
Podfile directory and uses the application's `PODS_PODFILE_DIR_PATH`, with the
argument quoted for paths containing spaces. It preserves the inherited flags
and the explicit C++ standard-library dependency. Apple public C declarations
also gain `used` with default visibility. Clang's [attribute documentation](https://github.com/llvm/llvm-project/blob/a031568275b4ec11f5bae714bf6b556358210b5b/clang/include/clang/Basic/AttrDocs.td)
specifies that `used` protects definitions from Mach-O section collection. The
[Apple linker source](https://github.com/apple-oss-distributions/ld64/blob/f60a74eaa2c99585de1dc0f2820e7a9f8aaf522c/src/ld/Options.cpp)
separately shows that ordinary executable globals do not all become dead-strip
roots by default. The change does not globally disable stripping or alter the
application's export restrictions.

## Bounded local compiler controls

These are **Linux-hosted Clang controls**, not Apple execution. Clang 19.1.7
compiled a minimal function using the actual public header for
`arm64-apple-ios15.0` without an SDK. The previous declaration did not emit a
`no_dead_strip` directive; the corrected declaration did. The complete
preprocessed public header was also byte-identical before and after the change
for six non-Apple combinations: Linux and Windows, each with static/default,
`CRISPEMBED_BUILD`, and `CRISPEMBED_SHARED` definitions.

| Receipt | Result | Interpretation |
|---|---:|---|
| [Initial probe](initial-probe.json) | 4/11 | Probe defects: compiler warnings included different before/after paths in the compared stream, and a broad function-name pattern matched a config typedef. The original result is retained. |
| [Corrected probe](corrected-probe.json) | 11/11 | Separate diagnostic/output streams and the actual public function inventory resolve those probe defects. Includes the Mach-O directive, all six byte comparisons, and generated dynamic-lookup source checks. |

The corrected receipt records checkout `9da573324e7fc15077bfa78daeecbed85f9fbebc`
plus exact modified source hashes. The new header SHA-256 is
`b179501794cc9092a8962ecae0c763ce016757f8ce66be5d0df727da6c9c4f93`.
`cmake/MarkerIdentity.cmake` hashes this header; a new build therefore has a new
resume identity. Earlier native platform receipts remain evidence for their
recorded source, not an execution claim for this header change.

## Actual Apple consumer gate

`tests/test_flutter_pod_resources.py` now builds an unsigned Release application
with dead stripping through both a direct path containing spaces and a
Flutter-style plugin symlink. The compiled consumer object must have no static
references to any CrispEmbed symbol. The app's installed notice bundle is read
back and checked against the canonical bytes in both cases.

On macOS the built app loads its bundled dylib, resolves all 17 Marker functions
dynamically, initializes a context and reads its idle progress. On iOS the proof
checks all 17 functions in the app's defined external symbols and dynamic export
trie. A separate iOS build removes only the final-app force-load argument and
must lose those dynamically named exports. That negative build retains its own
logs and does not modify the shipped podspec or input archive.

Actual Apple execution of this expanded gate is pending at the time of this
receipt. The iOS gate qualifies compile, link, resource bytes and export
retention; it does not execute on an iOS device. This native CocoaPods consumer
also does not qualify a Flutter SDK application. The separate Flutter lane
records its own SDK and app evidence.
