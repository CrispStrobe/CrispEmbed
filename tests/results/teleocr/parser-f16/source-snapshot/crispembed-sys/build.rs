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

/// Look for an already-built `libcrispembed` before considering a source build.
///
/// This deliberately takes `manifest_dir` rather than a resolved source root:
/// a prebuilt library is usable with NO C/C++ sources present at all, which is
/// the entire point of shipping one. Requiring `resolve_src_root` to succeed
/// first meant a consumer holding a perfectly good `libcrispembed.so` still had
/// to check out ~1 GB of sources plus the ggml submodule, or the build panicked
/// before it ever looked. v0.16.1 did not have that constraint.
fn try_prebuilt(manifest_dir: &Path) -> Option<PathBuf> {
    if let Ok(dir) = env::var("CRISPEMBED_SYS_LIB_DIR") {
        let path = PathBuf::from(dir);
        if has_prebuilt(&path) {
            return Some(path);
        }
        // Set but unusable. Say so — the alternative is a silent ~10-minute
        // source build (or a confusing "sources not found" panic) when the
        // user believed they had opted out of building entirely.
        println!(
            "cargo:warning=CRISPEMBED_SYS_LIB_DIR={} is set but contains no crispembed library \
             (looked for crispembed.lib, Release/crispembed.lib, libcrispembed.so, \
             libcrispembed.dylib) — ignoring it",
            path.display()
        );
    }

    // In-tree build directories, probed relative to the crate's parent (the
    // repo root for a checkout) and to the vendored copy. No source-tree
    // validation: `build/libcrispembed.so` is just as linkable whether or not
    // the ggml submodule was ever initialised.
    let roots = [
        manifest_dir.parent().map(Path::to_path_buf),
        Some(manifest_dir.join("vendor")),
    ];
    roots
        .into_iter()
        .flatten()
        .flat_map(|root| {
            [
                root.join("build-cuda"),
                root.join("build"),
                root.join("build-vulkan"),
            ]
        })
        .find(|path| has_prebuilt(path))
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
        // Only the shared library this crate links. The top-level CMakeLists
        // also declares the CLI, server, quantizer and ~40 test executables;
        // building all of them would cost consumers minutes for artifacts they
        // never use, and the published crate ships the test SOURCES (cmake
        // needs them to configure) without the fixtures that would let them run.
        .arg("--target")
        .arg("crispembed-shared")
        .arg("--config")
        .arg("Release");
    run(&mut build, "cmake build");

    build_dir
}

/// Locate the C/C++ sources cmake is pointed at.
///
/// In-tree they are the repository root (this crate's parent); in the crate
/// published to crates.io they are `vendor/`, because cargo only packages files
/// under the crate root. The repository copy wins when present, so a
/// development build never compiles a stale vendored snapshot.
///
/// Call this ONLY once a source build is known to be necessary — it panics when
/// the sources are absent, which is a perfectly normal state for a consumer
/// linking a prebuilt library.
fn resolve_src_root(manifest_dir: &Path) -> PathBuf {
    // Probe for CMakeLists.txt AND ggml, not just the directory: in a published
    // crate the parent is the cargo registry's `src/` directory, which exists
    // but holds no sources.
    let is_source_root =
        |p: &Path| p.join("CMakeLists.txt").is_file() && p.join("ggml/CMakeLists.txt").is_file();

    if let Some(repo) = manifest_dir.parent() {
        if is_source_root(repo) {
            return repo.to_path_buf();
        }
    }
    let vendored = manifest_dir.join("vendor");
    if is_source_root(&vendored) {
        return vendored;
    }
    panic!(
        "crispembed sources not found. Expected either the repository root \
         (in-tree build — did you run `git submodule update --init ggml`?) or \
         {}. For a checkout, run scripts/vendor_rust_sources.sh before \
         `cargo package`/`cargo publish`.",
        vendored.display()
    );
}

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    println!("cargo:rerun-if-env-changed=CRISPEMBED_SYS_LIB_DIR");

    // Prebuilt first, and only resolve the C/C++ sources if we actually have to
    // build them. Resolving unconditionally made `resolve_src_root`'s panic
    // reachable for consumers who had a perfectly good prebuilt library,
    // defeating both `CRISPEMBED_SYS_LIB_DIR` and the `build/` probe.
    let lib_dir = try_prebuilt(&manifest_dir)
        .unwrap_or_else(|| configure_and_build(&resolve_src_root(&manifest_dir)));
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
