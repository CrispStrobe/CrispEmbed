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
    if [ ! -f Libs/libcrispembed.dylib ]; then
      mkdir -p Libs
      url="https://github.com/CrispStrobe/CrispEmbed/releases/download/v#{s.version}/crispembed-macos-arm64.tar.gz"
      echo "crispembed: fetching prebuilt macOS libs -> $url"
      tmp=$(mktemp -d)
      curl -fsSL "$url" -o "$tmp/lib.tgz"
      tar -xzf "$tmp/lib.tgz" -C "$tmp"
      find "$tmp" -name '*.dylib' -exec cp -P {} Libs/ \\;
      rm -rf "$tmp"
    fi
  CMD

  s.vendored_libraries = 'Libs/*.dylib'

  # Ensure the dylibs are code-signed and embedded in the app bundle.
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'LD_RUNPATH_SEARCH_PATHS' => '$(inherited) @loader_path/../Frameworks',
  }
end
