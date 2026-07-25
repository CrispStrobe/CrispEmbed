Pod::Spec.new do |s|
  s.name             = 'crispembed'
  s.version          = '0.16.0'
  s.summary          = 'CrispEmbed on-device inference — embeddings + math OCR via ggml.'
  s.homepage         = 'https://github.com/CrispStrobe/CrispEmbed'
  s.license          = { :type => 'MIT' }
  s.author           = { 'CrispStrobe' => 'info@crispstrobe.com' }
  s.source           = { :path => '.' }

  s.platform         = :osx, '10.15'
  s.osx.deployment_target = '10.15'

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
    # Collapse each library to ONE real file named by its SONAME
    # (@rpath/libX.N.dylib). The prebuilt tarball ships versioned files
    # (libX.N.m.p.dylib) with libX.N.dylib only as a symlink, but CocoaPods
    # DEREFERENCES that symlink when it embeds the vendored libs — so the
    # @rpath/libX.N.dylib name every binary's LC_LOAD/LC_ID references would be
    # ABSENT from the app bundle (dyld "Library not loaded: @rpath/libX.N.dylib"
    # → SIGABRT at launch). Renaming the real file to its own install-name makes
    # the embedded filename match the load command, and removing the alias
    # symlinks drops the duplicate file references that also produced the
    # "malformed project / member of multiple groups" warnings. Idempotent.
    ( cd Libs
      for f in *.dylib; do
        [ -L "$f" ] && continue
        soname=$(basename "$(otool -D "$f" 2>/dev/null | tail -1)")
        case "$soname" in lib*.dylib) ;; *) continue ;; esac
        [ "$f" = "$soname" ] && continue
        rm -f "$soname"
        mv -f "$f" "$soname"
      done
      find . -type l -name '*.dylib' -delete
    )
  CMD

  s.vendored_libraries = 'Libs/*.dylib'

  # Ensure the dylibs are code-signed and embedded in the app bundle.
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'LD_RUNPATH_SEARCH_PATHS' => '$(inherited) @loader_path/../Frameworks',
  }
end
