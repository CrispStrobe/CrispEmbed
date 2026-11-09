#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an actual CocoaPods :path consumer of an unmodified CrispEmbed podspec.

macOS builds and launches an owned native app; iOS builds/links an unsigned
device-SDK app without claiming device execution. Both read the built notice
bundle. This is native pod integration, not Flutter SDK/app qualification.
Requires an Apple runner with Xcode and CocoaPods, plus this build's native libs.
No release download, model inference, signing identity or publication is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
NOTICES = {
    "NOTICE.marker": "NOTICE.marker",
    "NOTICE.teleocr": "NOTICE.teleocr",
    "Apache-2.0.txt": "models/licenses/apache-2.0/LICENSE",
}
APP = "CrispPodProbe"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_text(apple_platform):
    sdk = "macosx" if apple_platform == "macos" else "iphoneos"
    deployment = "MACOSX_DEPLOYMENT_TARGET = 12.0;" if apple_platform == "macos" else "IPHONEOS_DEPLOYMENT_TARGET = 15.0;"
    runpath = "@executable_path/../Frameworks" if apple_platform == "macos" else "@executable_path/Frameworks"
    # An ordinary application project with one Objective-C source. CocoaPods
    # performs the dependency, framework-embedding and resource integration.
    return '''// !$*UTF8*$!
{
 archiveVersion = 1;
 classes = {};
 objectVersion = 56;
 objects = {
  A00000000000000000000001 = {isa = PBXBuildFile; fileRef = A00000000000000000000002; };
  A00000000000000000000002 = {isa = PBXFileReference; lastKnownFileType = sourcecode.c.objc; path = main.m; sourceTree = "<group>"; };
  A00000000000000000000003 = {isa = PBXFileReference; explicitFileType = wrapper.application; includeInIndex = 0; path = CrispPodProbe.app; sourceTree = BUILT_PRODUCTS_DIR; };
  A00000000000000000000004 = {isa = PBXGroup; children = (A00000000000000000000002, A00000000000000000000005); sourceTree = "<group>"; };
  A00000000000000000000005 = {isa = PBXGroup; children = (A00000000000000000000003); name = Products; sourceTree = "<group>"; };
  A00000000000000000000006 = {isa = PBXSourcesBuildPhase; buildActionMask = 2147483647; files = (A00000000000000000000001); runOnlyForDeploymentPostprocessing = 0; };
  A00000000000000000000007 = {isa = PBXFrameworksBuildPhase; buildActionMask = 2147483647; files = (); runOnlyForDeploymentPostprocessing = 0; };
  A00000000000000000000008 = {isa = PBXResourcesBuildPhase; buildActionMask = 2147483647; files = (); runOnlyForDeploymentPostprocessing = 0; };
  A00000000000000000000009 = {isa = PBXNativeTarget; buildConfigurationList = A0000000000000000000000A; buildPhases = (A00000000000000000000006, A00000000000000000000007, A00000000000000000000008); buildRules = (); dependencies = (); name = CrispPodProbe; productName = CrispPodProbe; productReference = A00000000000000000000003; productType = "com.apple.product-type.application"; };
  A0000000000000000000000A = {isa = XCConfigurationList; buildConfigurations = (A0000000000000000000000B, A0000000000000000000000C); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release; };
  A0000000000000000000000B = {isa = XCBuildConfiguration; buildSettings = {__TARGET_SETTINGS__}; name = Debug; };
  A0000000000000000000000C = {isa = XCBuildConfiguration; buildSettings = {__TARGET_SETTINGS__}; name = Release; };
  A0000000000000000000000D = {isa = XCConfigurationList; buildConfigurations = (A0000000000000000000000E, A0000000000000000000000F); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release; };
  A0000000000000000000000E = {isa = XCBuildConfiguration; buildSettings = {CLANG_ENABLE_MODULES = YES; SDKROOT = __SDK__;}; name = Debug; };
  A0000000000000000000000F = {isa = XCBuildConfiguration; buildSettings = {CLANG_ENABLE_MODULES = YES; SDKROOT = __SDK__;}; name = Release; };
  A00000000000000000000010 = {isa = PBXProject; attributes = {LastUpgradeCheck = 1600;}; buildConfigurationList = A0000000000000000000000D; compatibilityVersion = "Xcode 14.0"; developmentRegion = en; hasScannedForEncodings = 0; knownRegions = (en, Base); mainGroup = A00000000000000000000004; productRefGroup = A00000000000000000000005; projectDirPath = ""; projectRoot = ""; targets = (A00000000000000000000009); };
 };
 rootObject = A00000000000000000000010;
}
'''.replace("__SDK__", sdk).replace("__TARGET_SETTINGS__", f'''
 ARCHS = arm64;
 CODE_SIGNING_ALLOWED = NO;
 CODE_SIGNING_REQUIRED = NO;
 ENABLE_USER_SCRIPT_SANDBOXING = NO;
 GENERATE_INFOPLIST_FILE = NO;
 HEADER_SEARCH_PATHS = ("$(inherited)", "$(PROJECT_DIR)/native");
 INFOPLIST_FILE = Info.plist;
 LD_RUNPATH_SEARCH_PATHS = ("$(inherited)", "{runpath}");
 OTHER_LDFLAGS = ("$(inherited)", "-framework", Foundation);
 PRODUCT_BUNDLE_IDENTIFIER = org.crispembed.ownedpodprobe;
 PRODUCT_NAME = "$(TARGET_NAME)";
 SDKROOT = {sdk};
 {deployment}
''')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("macos", "ios"), required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("this proof requires an actual Apple runner")
    native_dir, out = args.native_dir.resolve(strict=True), args.out.resolve()
    if out.exists():
        parser.error("--out must be a new directory; retain earlier attempts")
    out.mkdir(parents=True)
    checks, commands, toolchain = [], [], {}
    env = os.environ.copy()
    env.update(COCOAPODS_DISABLE_STATS="true", CRISPEMBED_OFFLINE="1")
    error = None
    selected = []
    bundle_hashes = {}

    def sanitized(value):
        for path, label in ((out, "<OUTPUT>"), (native_dir, "<NATIVE_LIBRARIES>"), (ROOT, "<REPOSITORY>")):
            value = value.replace(str(path), label)
        return re.sub(r"/Users/[^/\s]+", "<HOME>", value)

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)

    def run(label, command, cwd=ROOT, timeout=120, launch=False):
        command_env = env.copy()
        if launch:
            for key in ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "LD_LIBRARY_PATH"):
                command_env.pop(key, None)
        result = subprocess.run(command, cwd=cwd, env=command_env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=timeout)
        (out / (label + ".log")).write_text(sanitized(result.stdout), encoding="utf-8")
        commands.append({"label": label, "command": [sanitized(str(part)) for part in command],
                         "exit_code": result.returncode})
        check(label + ": command succeeds", result.returncode == 0)
        return result.stdout

    source_paths = ["tests/test_flutter_pod_resources.py", "src/crispembed.h", "flutter/crispembed/pubspec.yaml",
                    f"flutter/crispembed/{args.platform}/crispembed.podspec", *NOTICES.values()]
    source_paths += [f"flutter/crispembed/{args.platform}/Notices/{name}" for name in NOTICES]
    source_hashes = {name: sha(ROOT / name) for name in source_paths}
    expected = {name: (ROOT / source).read_bytes() for name, source in NOTICES.items()}
    try:
        toolchain["xcode"] = run("xcode-version", ["xcodebuild", "-version"]).strip()
        toolchain["cocoapods"] = run("cocoapods-version", ["pod", "--version"]).strip()
        sdk = "macosx" if args.platform == "macos" else "iphoneos"
        toolchain["sdk"] = run("sdk-version", ["xcrun", "--sdk", sdk, "--show-sdk-version"]).strip()
        plugin = out / "plugin"
        staged = plugin / args.platform
        shutil.copytree(ROOT / "flutter/crispembed" / args.platform, staged,
                        ignore=shutil.ignore_patterns("Libs", "Pods", "build"))
        for name in ("pubspec.yaml", "LICENSE", *NOTICES):
            shutil.copyfile(ROOT / "flutter/crispembed" / name, plugin / name)
        libs = staged / "Libs"
        libs.mkdir()
        candidates = sorted(native_dir.glob("*.dylib")) if args.platform == "macos" else [native_dir / "libcrispembed-static.a"]
        check("current native input is available", bool(candidates) and all(path.is_file() for path in candidates))
        if args.platform == "macos":
            check("macOS input uses a flat real-file dylib set", all(not path.is_symlink() for path in candidates) and
                  any(path.name == "libcrispembed.dylib" for path in candidates))
        for path in candidates:
            shutil.copyfile(path, libs / path.name)
            selected.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha(path)})
        for name, data in expected.items():
            check(name + ": shipped path-install resource is exact", (staged / "Notices" / name).read_bytes() == data)
        check("podspec is unmodified", sha(staged / "crispembed.podspec") == source_hashes[f"flutter/crispembed/{args.platform}/crispembed.podspec"])
        project = out / "consumer"
        project.mkdir()
        (project / "native").mkdir()
        shutil.copyfile(ROOT / "src/crispembed.h", project / "native/crispembed.h")
        (project / "main.m").write_text('''#import <Foundation/Foundation.h>
#include "crispembed.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
    @autoreleasepool {
        crispembed_marker_config config = crispembed_marker_default_config();
        if (config.struct_size != sizeof(config)) return 11;
        setenv("CRISPEMBED_MARKER", "1", 1);
        setenv("CRISPEMBED_TELEOCR", "0", 1);
        config.mode = 1;
        crispembed_marker_context * context = crispembed_marker_init(&config);
        if (!context) return 12;
        const char * progress = crispembed_marker_progress(context);
        int success = progress && strstr(progress, "\\\"state\\\":\\\"idle\\\"");
        crispembed_marker_free(context);
        if (!success) return 13;
        puts("Owned CocoaPods consumer: actual Marker ABI initialized successfully");
    }
    return 0;
}
''', encoding="utf-8")
        (project / "Info.plist").write_text('''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>org.crispembed.ownedpodprobe</string>
<key>CFBundleExecutable</key><string>$(EXECUTABLE_NAME)</string>
<key>CFBundleName</key><string>CrispPodProbe</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>1.0</string>
<key>CFBundleVersion</key><string>1</string>
</dict></plist>
''', encoding="utf-8")
        xcodeproj = project / (APP + ".xcodeproj")
        schemes = xcodeproj / "xcshareddata/xcschemes"
        schemes.mkdir(parents=True)
        (xcodeproj / "project.pbxproj").write_text(project_text(args.platform), encoding="utf-8")
        (schemes / (APP + ".xcscheme")).write_text('''<?xml version="1.0" encoding="UTF-8"?>
<Scheme LastUpgradeVersion="1600" version="1.3">
<BuildAction parallelizeBuildables="YES" buildImplicitDependencies="YES"><BuildActionEntries>
<BuildActionEntry buildForTesting="YES" buildForRunning="YES" buildForProfiling="YES" buildForArchiving="YES" buildForAnalyzing="YES">
<BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="A00000000000000000000009" BuildableName="CrispPodProbe.app" BlueprintName="CrispPodProbe" ReferencedContainer="container:CrispPodProbe.xcodeproj"/>
</BuildActionEntry></BuildActionEntries></BuildAction>
</Scheme>
''', encoding="utf-8")
        apple_platform, floor = ("osx", "12.0") if args.platform == "macos" else ("ios", "15.0")
        (project / "Podfile").write_text(f'''platform :{apple_platform}, '{floor}'
install! 'cocoapods', :deterministic_uuids => true
project '{APP}.xcodeproj'
target '{APP}' do
  pod 'crispembed', :path => '../plugin/{args.platform}'
end
''', encoding="utf-8")
        run("pod-install", ["pod", "install", "--verbose"], cwd=project)
        lock = (project / "Podfile.lock").read_text()
        check("CocoaPods resolves the real local path pod", "EXTERNAL SOURCES:" in lock and
              f":path: \"../plugin/{args.platform}\"" in lock or
              "EXTERNAL SOURCES:" in lock and f":path: ../plugin/{args.platform}" in lock)
        for name, data in expected.items():
            check(name + ": path install preserves shipped source bytes", (staged / "Notices" / name).read_bytes() == data)
        destination = "platform=macOS,arch=arm64" if args.platform == "macos" else "generic/platform=iOS"
        derived = out / "derived"
        run("xcodebuild", ["xcodebuild", "-workspace", str(project / (APP + ".xcworkspace")), "-scheme", APP,
                           "-configuration", "Debug", "-sdk", sdk, "-destination", destination,
                           "-derivedDataPath", str(derived), "CODE_SIGNING_ALLOWED=NO", "CODE_SIGNING_REQUIRED=NO",
                           "ENABLE_USER_SCRIPT_SANDBOXING=NO", "build"], cwd=project, timeout=240)
        product_dir = "Debug" if args.platform == "macos" else "Debug-iphoneos"
        app = derived / "Build/Products" / product_dir / (APP + ".app")
        check("unsigned native application was produced", app.is_dir())
        bundles = list(app.rglob("crispembed_native_notices.bundle"))
        check("the built application contains exactly one native notice bundle", len(bundles) == 1)
        for name, data in expected.items():
            files = list(bundles[0].rglob(name))
            check(name + ": actual built bundle preserves canonical bytes", len(files) == 1 and files[0].read_bytes() == data)
            bundle_hashes[name] = sha(files[0])
        if args.platform == "macos":
            check("CocoaPods embeds the real dylib inside the app", (app / "Contents/Frameworks/libcrispembed.dylib").is_file())
            output = run("native-app-launch", [str(app / "Contents/MacOS" / APP)], cwd=out, timeout=30, launch=True)
            check("bundled macOS application executes the actual Marker C ABI", "actual Marker ABI initialized successfully" in output)
        check("repository package inputs are unchanged", all(sha(ROOT / name) == digest for name, digest in source_hashes.items()))
        check("native input libraries are unchanged", all(sha(native_dir / item["name"]) == item["sha256"] for item in selected))
    except Exception as exc:
        error = sanitized(f"{type(exc).__name__}: {exc}")
    report = {
        "schema_version": 1, "kind": "actual-cocoapods-path-consumer", "platform": args.platform,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "host_architecture": platform.machine(), "toolchain": toolchain, "source_sha256": source_hashes,
        "native_libraries": selected, "bundle_notice_sha256": bundle_hashes, "commands": commands, "checks": checks,
        "passed": error is None and bool(checks) and all(row["passed"] for row in checks),
        "passed_count": sum(row["passed"] for row in checks), "total_count": len(checks),
        "macos_native_app_executed": args.platform == "macos" and any(row["name"].startswith("bundled macOS application") and row["passed"] for row in checks),
        "ios_device_runtime_qualified": False, "flutter_sdk_app_qualified": False,
        "limits": "Native CocoaPods path install, actual unsigned app link and resource-bundle bytes only. macOS additionally launches the app. No iOS device execution, Flutter SDK app, model inference, signing identity or publication.",
        "logs_sha256": {path.name: sha(path) for path in sorted(out.glob("*.log"))},
    }
    if error:
        report["error"] = error
    (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"CocoaPods {args.platform}: {report['passed_count']}/{report['total_count']} checks passed")
    if error:
        print(error, file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
