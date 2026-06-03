Pod::Spec.new do |s|
  s.name             = 'crispembed'
  s.version          = '0.3.0'
  s.summary          = 'CrispEmbed on-device inference — embeddings + math OCR via ggml.'
  s.homepage         = 'https://github.com/CrispStrobe/CrispEmbed'
  s.license          = { :type => 'MIT' }
  s.author           = { 'CrispStrobe' => 'info@crispstrobe.com' }
  s.source           = { :path => '.' }

  s.platform         = :osx, '10.15'
  s.osx.deployment_target = '10.15'

  # Pre-built shared library produced by CI (build-macos.sh / build.yml).
  # Place libcrispembed.dylib in macos/Libs/ before pod install.
  s.vendored_libraries = 'Libs/libcrispembed.dylib'

  # Ensure the dylib is code-signed and embedded in the app bundle.
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'LD_RUNPATH_SEARCH_PATHS' => '$(inherited) @loader_path/../Frameworks',
  }
end
