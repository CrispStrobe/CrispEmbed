use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src_root = manifest_dir.parent().expect("crispembed-sys must be inside the repo root");

    let mut cfg = cmake::Config::new(src_root);
    cfg.define("CRISPEMBED_BUILD_SHARED", "ON")
       .define("GGML_BLAS", "OFF")  // keep deps minimal by default
       .profile("Release");

    if cfg!(feature = "cuda") {
        cfg.define("GGML_CUDA", "ON");
    }
    if cfg!(feature = "metal") {
        cfg.define("GGML_METAL", "ON");
        cfg.define("GGML_METAL_EMBED_LIBRARY", "ON");
    }
    if cfg!(feature = "vulkan") {
        cfg.define("GGML_VULKAN", "ON");
    }

    let dst = cfg.build();

    // cmake crate puts build output in <dst>/build by default
    let lib_dir = dst.join("build");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    // Also check Release subdir (Windows MSVC multi-config)
    println!("cargo:rustc-link-search=native={}", lib_dir.join("Release").display());

    println!("cargo:rustc-link-lib=dylib=crispembed");

    // Platform C++ runtime
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "linux"   => println!("cargo:rustc-link-lib=dylib=stdc++"),
        "macos"   => println!("cargo:rustc-link-lib=dylib=c++"),
        _         => {}
    }
}
