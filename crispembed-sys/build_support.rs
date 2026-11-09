//! Target policy shared by the Cargo build script and model-free hosted tests.
//!
//! Presence of a correctly named file is a configuration check, not proof of
//! its architecture or dependency closure. Consumers must build/verify that
//! closure for TARGET; the target linker remains the final binary check.

use std::path::{Path, PathBuf};

const HOST_LIBRARIES: &[&str] = &[
    "crispembed.lib",
    "Release/crispembed.lib",
    "libcrispembed.so",
    "libcrispembed.dylib",
];
const IOS_LIBRARIES: &[&str] = &["libcrispembed.a"];
const ANDROID_LIBRARIES: &[&str] = &["libcrispembed.so"];
const IOS_LINKS: &[&str] = &[
    "dylib=c++",
    "framework=Foundation",
    "framework=Metal",
    "framework=MetalKit",
    "framework=Accelerate",
];

#[derive(Debug, PartialEq, Eq)]
pub struct LinkPlan {
    pub requires_explicit_prebuilt: bool,
    pub library_link: &'static str,
    pub extra_links: &'static [&'static str],
    pub library_files: &'static [&'static str],
}

pub fn link_plan(host: &str, target: &str, target_os: &str) -> Result<LinkPlan, String> {
    if host.is_empty() || target.is_empty() || target_os.is_empty() {
        return Err("Cargo HOST, TARGET and CARGO_CFG_TARGET_OS must be set".into());
    }
    let cross = host != target;
    let mobile = matches!(target_os, "ios" | "android");
    let extra_links: &[&str] = match target_os {
        "ios" => IOS_LINKS,
        // The verified Android shared-library closure includes static libc++.
        // Adding a host stdc++ or an unbundled c++_shared would break that closure.
        "android" => &[],
        "linux" => &["dylib=stdc++"],
        "macos" => &["dylib=c++"],
        _ => &[],
    };
    let library_files: &[&str] = match target_os {
        "ios" => IOS_LIBRARIES,
        "android" => ANDROID_LIBRARIES,
        _ if !cross => HOST_LIBRARIES,
        "linux" => &["libcrispembed.so"],
        "macos" => &["libcrispembed.dylib"],
        "windows" if target.ends_with("-gnu") => &["libcrispembed.dll.a", "crispembed.dll.a"],
        "windows" => &["crispembed.lib", "Release/crispembed.lib"],
        _ => {
            return Err(format!(
                "unsupported prebuilt CrispEmbed target {target} (OS {target_os}); \
                 automatic host CMake builds are disabled for cross compilation"
            ));
        }
    };
    Ok(LinkPlan {
        requires_explicit_prebuilt: cross || mobile,
        library_link: if target_os == "ios" {
            "static=crispembed"
        } else {
            "dylib=crispembed"
        },
        extra_links,
        library_files,
    })
}

pub fn find_prebuilt(
    plan: &LinkPlan,
    dir: &Path,
    mut is_file: impl FnMut(&Path) -> bool,
) -> Option<PathBuf> {
    plan.library_files
        .iter()
        .map(|name| dir.join(name))
        .find(|path| is_file(path))
}

/// Mobile/cross builds fail here before any host directory probe or CMake call.
/// Native desktop builds retain the existing warning-and-fallback behavior.
pub fn select_explicit_prebuilt(
    plan: &LinkPlan,
    configured_dir: Option<&Path>,
    is_file: impl FnMut(&Path) -> bool,
) -> Result<Option<PathBuf>, String> {
    if let Some(dir) = configured_dir.filter(|dir| !dir.as_os_str().is_empty()) {
        if find_prebuilt(plan, dir, is_file).is_some() {
            return Ok(Some(dir.to_path_buf()));
        }
        if plan.requires_explicit_prebuilt {
            return Err(format!(
                "CRISPEMBED_SYS_LIB_DIR={} contains no target library; expected {}. \
                 Supply a verified prebuilt closure for Cargo TARGET. \
                 Host directory probing and automatic CMake are disabled for mobile/cross builds",
                dir.display(),
                plan.library_files.join(" or ")
            ));
        }
    }
    if plan.requires_explicit_prebuilt {
        return Err(format!(
            "CRISPEMBED_SYS_LIB_DIR must explicitly name a target prebuilt directory containing {}. \
             Host directory probing and automatic CMake are disabled for mobile/cross builds",
            plan.library_files.join(" or ")
        ));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ios_device_and_simulators_link_complete_static_closure() {
        for target in [
            "aarch64-apple-ios",
            "aarch64-apple-ios-sim",
            "x86_64-apple-ios",
        ] {
            let plan = link_plan("aarch64-apple-darwin", target, "ios").unwrap();
            assert!(plan.requires_explicit_prebuilt);
            assert_eq!(plan.library_link, "static=crispembed");
            assert_eq!(plan.library_files, &["libcrispembed.a"]);
            assert_eq!(plan.extra_links, IOS_LINKS);
            assert_eq!(
                select_explicit_prebuilt(&plan, Some(Path::new("ios/lib")), |p| {
                    p == Path::new("ios/lib/libcrispembed.a")
                })
                .unwrap(),
                Some(PathBuf::from("ios/lib"))
            );
            assert!(select_explicit_prebuilt(&plan, Some(Path::new("host/build")), |p| {
                p.ends_with("libcrispembed.dylib")
            })
            .is_err());
        }
    }

    #[test]
    fn android_links_only_the_supplied_shared_closure() {
        for target in [
            "aarch64-linux-android",
            "armv7-linux-androideabi",
            "x86_64-linux-android",
        ] {
            let plan = link_plan("x86_64-unknown-linux-gnu", target, "android").unwrap();
            assert!(plan.requires_explicit_prebuilt);
            assert_eq!(plan.library_link, "dylib=crispembed");
            assert!(plan.extra_links.is_empty());
            assert_eq!(plan.library_files, &["libcrispembed.so"]);
            assert!(select_explicit_prebuilt(&plan, None, |_| true).is_err());
            assert!(select_explicit_prebuilt(&plan, Some(Path::new("")), |_| true).is_err());
            assert!(select_explicit_prebuilt(&plan, Some(Path::new("android/lib")), |p| {
                p.ends_with("libcrispembed.a")
            })
            .is_err());
        }
    }

    #[test]
    fn mobile_never_uses_host_fallback_even_when_host_equals_target() {
        for (target, os) in [
            ("aarch64-apple-ios", "ios"),
            ("aarch64-linux-android", "android"),
        ] {
            let plan = link_plan(target, target, os).unwrap();
            assert!(select_explicit_prebuilt(&plan, None, |_| true).is_err());
        }
    }

    #[test]
    fn desktop_cross_targets_require_an_explicit_target_library() {
        for (target, os, library) in [
            ("aarch64-unknown-linux-gnu", "linux", "libcrispembed.so"),
            ("aarch64-apple-darwin", "macos", "libcrispembed.dylib"),
            ("x86_64-pc-windows-msvc", "windows", "crispembed.lib"),
            ("x86_64-pc-windows-gnu", "windows", "libcrispembed.dll.a"),
        ] {
            let plan = link_plan("x86_64-unknown-linux-gnu", target, os).unwrap();
            assert!(plan.requires_explicit_prebuilt);
            assert_eq!(plan.library_files[0], library);
            assert!(select_explicit_prebuilt(&plan, None, |_| true).is_err());
            assert!(select_explicit_prebuilt(&plan, Some(Path::new("missing")), |_| false).is_err());
            assert_eq!(
                select_explicit_prebuilt(&plan, Some(Path::new("target/lib")), |p| {
                    p == Path::new("target/lib").join(library)
                })
                .unwrap(),
                Some(PathBuf::from("target/lib"))
            );
        }
    }

    #[test]
    fn native_desktop_keeps_existing_fallback_and_cpp_runtime() {
        for (host, os, runtime) in [
            ("x86_64-unknown-linux-gnu", "linux", Some("dylib=stdc++")),
            ("aarch64-apple-darwin", "macos", Some("dylib=c++")),
            ("x86_64-pc-windows-msvc", "windows", None),
        ] {
            let plan = link_plan(host, host, os).unwrap();
            assert!(!plan.requires_explicit_prebuilt);
            assert_eq!(plan.library_link, "dylib=crispembed");
            assert_eq!(plan.library_files, HOST_LIBRARIES);
            assert_eq!(plan.extra_links.first().copied(), runtime);
            assert_eq!(select_explicit_prebuilt(&plan, None, |_| false).unwrap(), None);
            assert_eq!(
                select_explicit_prebuilt(&plan, Some(Path::new("missing")), |_| false).unwrap(),
                None
            );
            assert_eq!(
                find_prebuilt(&plan, Path::new("build"), |p| {
                    p == Path::new("build/Release/crispembed.lib")
                }),
                Some(PathBuf::from("build/Release/crispembed.lib"))
            );
        }
    }

    #[test]
    fn unknown_cross_targets_and_incomplete_cargo_environment_fail_clearly() {
        assert!(link_plan("x86_64-unknown-linux-gnu", "wasm32-unknown-unknown", "unknown").is_err());
        for (host, target, os) in [
            ("", "target", "ios"),
            ("host", "", "ios"),
            ("host", "target", ""),
        ] {
            assert!(link_plan(host, target, os).is_err());
        }
    }
}
