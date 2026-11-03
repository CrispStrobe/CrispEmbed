# The pod version doubles as the release tag the prebuilt libs are fetched
# from (see prepare_command below), so a hardcoded literal here silently
# pins every `pod install` to an old release: it sat at 0.16.0 while the
# project shipped 0.17.0/0.17.1/0.17.2, and macOS/iOS kept downloading the
# July libs. Derive it from the plugin's pubspec.yaml — the one file that is
# always shipped next to this podspec, on pub.dev and in a repo checkout
# alike — so bumping the release bumps this automatically.
Pod::Spec.new do |s|
  s.name             = 'crispembed'
  s.version          = File.read(File.join(__dir__, '..', 'pubspec.yaml'))[/^version:\s*([0-9][^\s+]*)/, 1]
  s.summary          = 'CrispEmbed on-device inference — embeddings + math OCR via ggml.'
  s.homepage         = 'https://github.com/CrispStrobe/CrispEmbed'
  s.license          = { :type => 'MIT' }
  s.author           = { 'CrispStrobe' => 'info@crispstrobe.com' }
  s.source           = { :path => '.' }

  # The distributed dylibs are Apple-silicon-only, and ggml's Metal event
  # synchronization calls an API introduced in macOS 12 — so 12.0 is the
  # real floor. Claiming 10.15, as this did, made the pod installable on
  # systems where the dylib cannot load at all.
  s.platform         = :osx, '12.0'
  s.osx.deployment_target = '12.0'

  # The prebuilt libs are produced by CI (release.yml) and published as GitHub
  # release assets; this fetches the tarball for this pod's version on `pod
  # install` (skipped if already present — e.g. a local dev drop). The tarball
  # bundles libcrispembed.dylib AND its libggml*.dylib siblings (the dylib is NOT
  # self-contained), so all of them are vendored and embedded.
  s.prepare_command = <<-CMD
    set -e
    mkdir -p Libs
    if ! ls Libs/*.dylib >/dev/null 2>&1; then
      url="https://github.com/CrispStrobe/CrispEmbed/releases/download/v#{s.version}/crispembed-macos-arm64.tar.gz"
      echo "crispembed: fetching prebuilt macOS libs -> $url"
      tmp=$(mktemp -d)
      curl -fsSL "$url" -o "$tmp/lib.tgz"
      tar -xzf "$tmp/lib.tgz" -C "$tmp"
      find "$tmp" -name '*.dylib' -exec cp -P {} Libs/ \\;
      rm -rf "$tmp"
    fi
    # One real file per library, named AND identified by its unversioned name
    # (@rpath/libX.dylib), with every cross-reference rewritten to match — see
    # flatten_dylibs.sh. CMake's SOVERSION gives the libs install names
    # @rpath/libX.0.dylib and ships the other names as symlinks; CocoaPods
    # dereferences or drops symlinks when it embeds vendored_libraries, so the
    # app bundle never held both the libcrispembed.dylib the Dart side dlopen()s
    # and the libX.0.dylib names the load commands ask for (dyld "Library not
    # loaded" at launch). The release tarball is already flat; this also covers
    # local drops of a CMake build/install tree. Idempotent; fails the install
    # if any @rpath reference would be left unresolved.
    sh flatten_dylibs.sh Libs
  CMD

  s.vendored_libraries = 'Libs/*.dylib'

  # Ensure the dylibs are code-signed and embedded in the app bundle.
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'LD_RUNPATH_SEARCH_PATHS' => '$(inherited) @loader_path/../Frameworks',
  }
end
