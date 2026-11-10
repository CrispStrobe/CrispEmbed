# PDF Craft actual Apple and Windows platform evidence

The [receipt](receipt.json) reads completed jobs from [Build 42607873214](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42607873214). The overall workflow still had queued Linux/Android siblings; the Apple and Windows results below were already complete.

The checkout was PR merge `e827fb55e421043be6880f51b4d55c387076ece6`. Its entire Git tree equals feature head `b34fe99611a6168f572630904531c79b02477b9f`, tree `03c70c5e0e6d2a9617852ab82d7c02074f274e46`. Runtime, relevant native fixtures and Apple podspec bytes also match runtime source `891ea3d3632c6a3c3155e25b34705da6a02ec807`; the receipt records exact scopes and hashes.

| Actual platform | Compiler | Executed Marker results |
| --- | --- | --- |
| Windows Server 2025 x86_64 | MSVC 19.51.36260.0 | Document 185, artifact 82, lifecycle 471; zero failures |
| macOS 26.7.2 arm64 | AppleClang 21.0.0.21000101 | Document 185, artifact 87, lifecycle 480, C API 74, save 38, backend 18; zero failures |
| iOS arm64 device SDK, on macOS | AppleClang 21.0.0.21000101 | Static native compilation; no device execution |

The macOS job also successfully generated with Xcode, passed both podspec Ruby syntax checks, installed/flatted the dylib set and loaded the resulting library. Those steps do not establish actual CocoaPods installation or Flutter app behavior. The follow-up consumer records those boundaries separately. Windows omits POSIX-specific filesystem controls, so its counts are intentionally lower.

## Negative package readback retained

All three downloaded artifacts match GitHub's advertised archive SHA-256. Their full member hashes are recorded. The desktop upload included `dist/lib` and `dist/bin` but omitted CMake's installed license directory; iOS uploaded only `libcrispembed-static.a`. Consequently **none contains `NOTICE.marker`, `NOTICE.teleocr` or `Apache-2.0.txt`**. The original iOS development archive also lacks its separately built static dependency archives, unlike the complete release package.

These findings require corrected development-package assembly and readback. They do not negate the completed native test results. The fix adds exact canonical notices, closes the iOS static archive dependencies and executes an actual native CocoaPods `:path` consumer. No corrected-package result is claimed by this pre-fix receipt.

The compact [Windows](windows-excerpts.log), [macOS](macos-excerpts.log) and [iOS](ios-excerpts.log) excerpts preserve checkout, compiler and assertion evidence. Full step results and decoded full-log hashes remain in the receipt. Binaries, archives, host paths and access details are excluded from Git.
