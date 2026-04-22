use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn print_link_info(lib_dir: &Path) {
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!(
        "cargo:rustc-link-search=native={}",
        lib_dir.join("Release").display()
    );
    println!("cargo:rustc-link-lib=dylib=crispembed");

    match env::var("CARGO_CFG_TARGET_OS").unwrap_or_default().as_str() {
        "linux" => println!("cargo:rustc-link-lib=dylib=stdc++"),
        "macos" => println!("cargo:rustc-link-lib=dylib=c++"),
        _ => {}
    }
}

fn has_prebuilt(dir: &Path) -> bool {
    dir.join("crispembed.lib").exists()
        || dir.join("Release").join("crispembed.lib").exists()
        || dir.join("libcrispembed.so").exists()
        || dir.join("libcrispembed.dylib").exists()
}

fn try_prebuilt(src_root: &Path) -> Option<PathBuf> {
    if let Ok(dir) = env::var("CRISPEMBED_SYS_LIB_DIR") {
        let path = PathBuf::from(dir);
        if has_prebuilt(&path) {
            return Some(path);
        }
    }

    let candidates = [
        src_root.join("build-cuda"),
        src_root.join("build"),
        src_root.join("build-vulkan"),
    ];
    candidates.into_iter().find(|path| has_prebuilt(path))
}

fn run(cmd: &mut Command, what: &str) {
    let status = cmd.status().unwrap_or_else(|err| {
        panic!("failed to start {what}: {err}");
    });
    if !status.success() {
        panic!("{what} failed with status {status}");
    }
}

fn configure_and_build(src_root: &Path) -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let build_dir = out_dir.join("crispembed-build");

    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(src_root)
        .arg("-B")
        .arg(&build_dir)
        .arg("-DCRISPEMBED_BUILD_SHARED=ON")
        .arg("-DGGML_BLAS=OFF")
        .arg("-DCMAKE_BUILD_TYPE=Release");

    if cfg!(feature = "cuda") {
        configure.arg("-DGGML_CUDA=ON");
    }
    if cfg!(feature = "metal") {
        configure.arg("-DGGML_METAL=ON");
        configure.arg("-DGGML_METAL_EMBED_LIBRARY=ON");
    }
    if cfg!(feature = "vulkan") {
        configure.arg("-DGGML_VULKAN=ON");
    }

    run(&mut configure, "cmake configure");

    let mut build = Command::new("cmake");
    build.arg("--build").arg(&build_dir).arg("--config").arg("Release");
    run(&mut build, "cmake build");

    build_dir
}

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src_root = manifest_dir
        .parent()
        .expect("crispembed-sys must be inside the repo root");

    println!("cargo:rerun-if-env-changed=CRISPEMBED_SYS_LIB_DIR");

    let lib_dir = try_prebuilt(src_root).unwrap_or_else(|| configure_and_build(src_root));
    print_link_info(&lib_dir);
}
