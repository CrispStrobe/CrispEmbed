Pod::Spec.new do |s|
  s.name             = 'crispembed'
  s.version          = '0.3.0'
  s.summary          = 'CrispEmbed on-device inference — embeddings + math OCR via ggml.'
  s.homepage         = 'https://github.com/CrispStrobe/CrispEmbed'
  s.license          = { :type => 'MIT' }
  s.author           = { 'CrispStrobe' => 'info@crispstrobe.com' }
  s.source           = { :path => '.' }

  s.platform         = :ios, '15.0'
  s.ios.deployment_target = '15.0'

  # Pre-built static library produced by CI (build.yml, ios-arm64 job).
  # Static linking: symbols become part of the main binary, loaded via
  # DynamicLibrary.process() in Dart.
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
