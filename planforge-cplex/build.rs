use std::env;
use std::path::{Path, PathBuf};

fn candidate_roots() -> Vec<PathBuf> {
    ["CPLEX_ROOT", "DOWNWARD_CPLEX_ROOT"]
        .into_iter()
        .filter_map(|name| env::var_os(name).map(PathBuf::from))
        .collect()
}

fn require_file(path: &Path, description: &str) {
    assert!(
        path.is_file(),
        "{description} not found at {}; set CPLEX_ROOT to the CPLEX directory \
         inside an unrestricted IBM ILOG CPLEX Studio installation",
        path.display()
    );
}

fn main() {
    println!("cargo:rerun-if-env-changed=CPLEX_ROOT");
    println!("cargo:rerun-if-env-changed=DOWNWARD_CPLEX_ROOT");
    println!("cargo:rerun-if-env-changed=DOCS_RS");

    if env::var_os("CARGO_FEATURE_CPLEX").is_none() || env::var_os("DOCS_RS").is_some() {
        return;
    }

    let roots = candidate_roots();
    let root = roots
        .iter()
        .find(|root| root.join("include/ilcplex/cplex.h").is_file())
        .unwrap_or_else(|| {
            panic!(
                "CPLEX was requested but no installation was found; set \
                 CPLEX_ROOT (preferred) or DOWNWARD_CPLEX_ROOT to the CPLEX \
                 directory, for example /opt/ibm/ILOG/CPLEX_Studio2211/cplex"
            )
        });

    require_file(&root.join("include/ilcplex/cplex.h"), "CPLEX C API header");

    let library_dir = root.join("lib/x86-64_linux/static_pic");
    let static_library = library_dir.join("libcplex.a");
    require_file(&static_library, "position-independent static CPLEX library");

    println!("cargo:rustc-link-search=native={}", library_dir.display());
    println!("cargo:rustc-link-lib=static=cplex");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=m");
}
