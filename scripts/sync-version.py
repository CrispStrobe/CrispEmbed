import sys
import re
import os

# Propagates the version string in the top-level VERSION file to every
# language binding's manifest. VERSION is the single source of truth (CMake
# also reads it directly). Mirrors CrispASR's scripts/sync-version.py, retargeted
# to CrispEmbed's crates/packages.


def update_file(file_path, patterns, version):
    if not os.path.exists(file_path):
        print(f"Skipping {file_path} (not found)")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = content
    for pattern, replacement in patterns:
        new_content = re.sub(pattern, replacement.replace("{version}", version), new_content, flags=re.MULTILINE)

    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {file_path}")
    else:
        print(f"No changes for {file_path}")


if __name__ == "__main__":
    version_file = 'VERSION'
    if not os.path.exists(version_file):
        print(f"Error: {version_file} file not found")
        sys.exit(1)

    with open(version_file, 'r', encoding='utf-8') as f:
        version = f.read().strip()

    print(f"Synchronizing version to {version}...")

    # Rust
    update_file('crispembed/Cargo.toml', [
        (r'^version = "[^"]+"', 'version = "{version}"'),
        (r'crispembed-sys = \{ path = "\.\./crispembed-sys", version = "[^"]+" \}', 'crispembed-sys = { path = "../crispembed-sys", version = "{version}" }')
    ], version)
    update_file('crispembed-sys/Cargo.toml', [
        (r'^version = "[^"]+"', 'version = "{version}"')
    ], version)

    # Python
    update_file('python/pyproject.toml', [
        (r'^version = "[^"]+"', 'version = "{version}"')
    ], version)

    # Dart/Flutter
    update_file('flutter/crispembed/pubspec.yaml', [
        (r'^version: [^\n]+', 'version: {version}')
    ], version)

    # JavaScript / Bindings (skipped gracefully if CrispEmbed has no JS package)
    update_file('bindings/javascript/package.json', [
        (r'^  "version": "[^"]+"', '  "version": "{version}"')
    ], version)

    print("Version synchronization complete.")
