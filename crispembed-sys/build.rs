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

/// Emit `cargo:rustc-link-arg=-Wl,-rpath,...` so binaries built from this
/// crate (and any reverse-dep crate that links it) can find
/// `libcrispembed` plus its ggml siblings at runtime.
///
/// On macOS the dylib's install name is `@rpath/libcrispembed.0.dylib`, so
/// without a resolving LC_RPATH on the consumer even `cargo run` fails
/// with "no LC_RPATH's found".
///
/// Three rpath entries are added on each Unix:
///   * absolute path to the freshly built lib dir — lets `cargo run` and
///     `cargo test` work directly from the workspace.
///   * absolute path to `<build>/ggml/src` — same, for the ggml siblings.
///   * `@executable_path/../Frameworks` / `$ORIGIN/../lib` — relative entry
///     so an end-user binary works once shipped (Tauri bundle, Debian
///     package, etc.). The `$ORIGIN` literal must reach the linker
///     un-substituted, so cargo's pass-through is exactly what we need.
fn emit_runtime_rpath(lib_dir: &Path) {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let lib_dir_str = lib_dir.display();
    // ggml siblings: when consuming a freshly-built tree, they live under
    // `<lib_dir>/ggml/src`; when consuming an installed prefix, they live
    // alongside libcrispembed (so the lib_dir itself works).
    let ggml_dir = lib_dir.join("ggml").join("src");
    let ggml_dir_str = ggml_dir.display();
    match target_os.as_str() {
        "macos" => {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir_str}");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{ggml_dir_str}");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../Frameworks");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../Frameworks");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../lib");
        }
        "linux" => {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir_str}");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{ggml_dir_str}");
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib");
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
        }
        _ => {} // Windows: DLL search path includes the exe's directory.
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
    build
        .arg("--build")
        .arg(&build_dir)
        .arg("--config")
        .arg("Release");
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
    emit_runtime_rpath(&lib_dir);

    // Publish the resolved lib dir on Cargo's `links = "crispembed"`
    // metadata channel so direct dependents see `DEP_CRISPEMBED_LIB_DIR`
    // and can emit additional `cargo:rustc-link-arg=-Wl,-rpath,…` against
    // it if they need to. Cargo only forwards links metadata to immediate
    // dependents — this crate already emits the most common rpath entries
    // via `emit_runtime_rpath`, but consumers can layer more on top.
    println!("cargo:LIB_DIR={}", lib_dir.display());
}
