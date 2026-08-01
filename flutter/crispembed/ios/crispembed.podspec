Pod::Spec.new do |s|
  s.name             = 'crispembed'
  s.version          = '0.16.0'
  s.summary          = 'CrispEmbed on-device inference — embeddings + math OCR via ggml.'
  s.homepage         = 'https://github.com/CrispStrobe/CrispEmbed'
  s.license          = { :type => 'MIT' }
  s.author           = { 'CrispStrobe' => 'info@crispstrobe.com' }
  s.source           = { :path => '.' }

  s.platform         = :ios, '15.0'
  s.ios.deployment_target = '15.0'

  # The static lib is produced by CI (release.yml, ios-arm64 job) and published
  # as a release asset; fetched for this pod's version on `pod install` (skipped
  # if already present). Static linking: symbols become part of the main binary,
  # loaded via DynamicLibrary.process() in Dart. The .a archives ggml, so it is
  # self-contained (no sibling libs needed).
  s.prepare_command = <<-CMD
    set -e
    if [ ! -f Libs/libcrispembed-static.a ]; then
      mkdir -p Libs
      url="https://github.com/CrispStrobe/CrispEmbed/releases/download/v#{s.version}/crispembed-ios-arm64.tar.gz"
      echo "crispembed: fetching prebuilt iOS lib -> $url"
      tmp=$(mktemp -d)
      curl -fsSL "$url" -o "$tmp/lib.tgz"
      tar -xzf "$tmp/lib.tgz" -C "$tmp"
      find "$tmp" -name 'libcrispembed-static.a' -exec cp {} Libs/ \\;
      rm -rf "$tmp"
    fi
  CMD

  s.vendored_libraries = 'Libs/libcrispembed-static.a'

  # Metal framework for GPU-accelerated ggml ops.
  s.frameworks = 'Accelerate', 'Metal', 'MetalKit'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    # Force-load the static archive so all C symbols are visible to Dart FFI.
    'OTHER_LDFLAGS' => '-force_load $(PODS_TARGET_SRCROOT)/Libs/libcrispembed-static.a',
  }

  # Dummy source so CocoaPods doesn't complain about missing source_files.
  s.source_files = 'Classes/**/*'
end
