# Derived, not hardcoded — see the note in macos/crispembed.podspec: this
# version is also the release tag prepare_command fetches the prebuilt libs
# from, and the literal had drifted three releases behind.
Pod::Spec.new do |s|
  s.name             = 'crispembed'
  s.version          = File.read(File.join(__dir__, '..', 'pubspec.yaml'))[/^version:\s*([0-9][^\s+]*)/, 1]
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
    mkdir -p Libs Notices
    if [ ! -f Libs/libcrispembed-static.a ]; then
      mkdir -p Libs
      # CRISPEMBED_PREBUILT_VERSION (opt-in) fetches the libs of another published release
      # tag, e.g. when this checkout's pubspec version has no release yet; unset = pod version.
      ver="${CRISPEMBED_PREBUILT_VERSION:-#{s.version}}"
      url="https://github.com/CrispStrobe/CrispEmbed/releases/download/v$ver/crispembed-ios-arm64.tar.gz"
      echo "crispembed: fetching prebuilt iOS lib -> $url"
      tmp=$(mktemp -d)
      curl -fsSL "$url" -o "$tmp/lib.tgz"
      tar -xzf "$tmp/lib.tgz" -C "$tmp"
      find "$tmp" -name 'libcrispembed-static.a' -exec cp {} Libs/ \\;
      for notice in NOTICE.marker NOTICE.teleocr Apache-2.0.txt; do
        if [ -f "$tmp/$notice" ]; then
          cp "$tmp/$notice" "Notices/$notice"
        fi
      done
      rm -rf "$tmp"
    fi
    # Local binary drops and older archives use the notices shipped with this
    # plugin. A fetched archive's exact notices take precedence when present.
    for notice in NOTICE.marker NOTICE.teleocr Apache-2.0.txt; do
      if [ ! -f "Notices/$notice" ]; then
        cp "../$notice" "Notices/$notice"
      fi
    done
  CMD

  s.vendored_libraries = 'Libs/libcrispembed-static.a'
  # These files ship with the plugin: :path installs skip prepare_command.
  s.resource_bundles = { 'crispembed_native_notices' => ['Notices/*'] }

  # Metal framework for GPU-accelerated ggml ops.
  s.frameworks = 'Accelerate', 'Metal', 'MetalKit'
  # The complete native archive contains C++; Objective-C/Swift consumers must
  # link its standard library even when they only call the public C ABI.
  s.libraries = 'c++'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
  }

  # FFI-only calls do not create link-time references. Force-load the archive in
  # the final application, not the vendored-only pod target. PODS_TARGET_SRCROOT
  # is unavailable there; resolve this pod's actual location relative to the
  # Podfile so direct :path installs and Flutter's plugin symlinks both work.
  # The public Apple C ABI uses `used` to survive Release dead stripping without
  # disabling dead stripping or changing export rules for the rest of the app.
  archive = Pathname.new(File.expand_path('Libs/libcrispembed-static.a', __dir__))
  archive_relative = archive.relative_path_from(Pod::Config.instance.installation_root)
  # Xcodeproj shell-splits OTHER_LDFLAGS and re-emits -force_load paths
  # unquoted, so a plain "..." pair is lost and a path with spaces breaks the
  # link. Single quotes keep the double quotes inside the token it re-emits.
  s.user_target_xcconfig = {
    'OTHER_LDFLAGS' => %($(inherited) -force_load '"$(PODS_PODFILE_DIR_PATH)/#{archive_relative}"'),
  }

  # This is a vendored-library pod; no dummy source file is required.
end
