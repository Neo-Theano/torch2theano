use crate::analyzer;
use crate::codegen;
use crate::report::{self, FileReport, Report};
use crate::transforms;
use anyhow::{Context, Result};
use colored::*;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

/// Directories that should be skipped during conversion.
const SKIP_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".eggs",
    "*.egg-info",
    "node_modules",
    ".venv",
    "venv",
    "env",
];

/// Dependency manifest filenames that get special treatment.
const DEP_FILES: &[&str] = &[
    "requirements.txt",
    "requirements-dev.txt",
    "requirements_dev.txt",
    "dev-requirements.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
];

/// Convert an entire directory tree from PyTorch Python to a Neo Theano Rust project.
pub fn convert_directory(
    input: &Path,
    output: &Path,
    dry_run: bool,
    verbose: bool,
) -> Result<Report> {
    let mut report = Report::new();

    // Phase 1: Collect and analyze all Python files
    let mut py_files: Vec<(String, String)> = Vec::new(); // (relative_path, source)
    let mut dep_files: Vec<(String, String)> = Vec::new();
    let mut other_files: Vec<(String, std::path::PathBuf)> = Vec::new();

    for entry in WalkDir::new(input)
        .into_iter()
        .filter_entry(|e| !should_skip(e))
    {
        let entry = entry?;
        let src_path = entry.path();

        if entry.file_type().is_dir() {
            continue;
        }

        report.files_scanned += 1;
        let rel = src_path.strip_prefix(input).unwrap_or(src_path);
        let rel_str = rel.display().to_string();
        let filename = src_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        if is_python_file(src_path) {
            let source = fs::read_to_string(src_path)
                .with_context(|| format!("reading {}", src_path.display()))?;
            py_files.push((rel_str, source));
        } else if is_dep_file(&filename) {
            let source = fs::read_to_string(src_path)
                .with_context(|| format!("reading {}", src_path.display()))?;
            dep_files.push((rel_str, source));
        } else if !is_binary_or_large(src_path) {
            other_files.push((rel_str, src_path.to_path_buf()));
        } else {
            report.files_skipped += 1;
        }
    }

    // Phase 2: Analyze Python files
    let analyzed: Vec<(String, analyzer::PyFile)> = py_files
        .iter()
        .map(|(path, source)| {
            let pyfile = analyzer::analyze(source);
            (path.clone(), pyfile)
        })
        .collect();

    // Phase 3: Generate Rust project
    // Derive project name from input directory name
    let project_name = input
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "converted_project".to_string());
    let codegen_result = codegen::generate_project_named(&analyzed, &project_name);

    // Record file reports for the codegen changes
    let fr = FileReport {
        path: "src/main.rs".to_string(),
        changes: codegen_result.changes,
    };

    if fr.change_count() > 0 || fr.warning_count() > 0 {
        report.files_transformed += py_files.len();
        if verbose {
            report::print_file_detail(&fr);
        }
    }

    report.file_reports.push(fr);

    // Also process dependency files for warnings
    for (rel_path, source) in &dep_files {
        let result = transforms::transform_dependency_file(source, rel_path);
        let fr = FileReport {
            path: rel_path.clone(),
            changes: result.changes,
        };
        if fr.change_count() > 0 || fr.warning_count() > 0 {
            report.files_transformed += 1;
            if verbose {
                report::print_file_detail(&fr);
            }
        }
        report.file_reports.push(fr);
    }

    // Phase 4: Write output
    if !dry_run {
        // Create output directory structure
        let src_dir = output.join("src");
        fs::create_dir_all(&src_dir)
            .with_context(|| format!("creating dir {}", src_dir.display()))?;

        // Write Cargo.toml
        let cargo_path = output.join("Cargo.toml");
        fs::write(&cargo_path, &codegen_result.cargo_toml)
            .with_context(|| format!("writing {}", cargo_path.display()))?;

        if verbose {
            println!(
                "  {} {}",
                "(generated)".dimmed(),
                "Cargo.toml".bold()
            );
        }

        // Write src/main.rs
        let main_rs_path = src_dir.join("main.rs");
        fs::write(&main_rs_path, &codegen_result.main_rs)
            .with_context(|| format!("writing {}", main_rs_path.display()))?;

        if verbose {
            for (rel_path, _) in &py_files {
                println!(
                    "  {} -> {}",
                    rel_path.dimmed(),
                    "src/main.rs".bold()
                );
            }
        }

        // Copy non-Python, non-binary files (README, configs, etc.)
        for (rel_path, src_path) in &other_files {
            let dst_path = output.join(rel_path);
            if let Some(parent) = dst_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(src_path, &dst_path)
                .with_context(|| {
                    format!("copying {} -> {}", src_path.display(), dst_path.display())
                })?;
            report.files_copied += 1;
        }

        println!(
            "  {} written to {}",
            "output".green(),
            output.display().to_string().bold()
        );
    }

    Ok(report)
}

// ── helpers ──────────────────────────────────────────────────────────

fn should_skip(entry: &walkdir::DirEntry) -> bool {
    let name = entry.file_name().to_string_lossy();
    if entry.file_type().is_dir() {
        return SKIP_DIRS.iter().any(|&d| {
            if d.starts_with('*') {
                name.ends_with(&d[1..])
            } else {
                name == d
            }
        });
    }
    false
}

fn is_python_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("py" | "pyi")
    )
}

fn is_dep_file(filename: &str) -> bool {
    DEP_FILES.iter().any(|&d| filename == d)
}

fn is_binary_or_large(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    matches!(
        ext,
        "pth" | "pt" | "bin" | "onnx" | "pkl" | "pickle"
            | "h5" | "hdf5" | "npz" | "npy"
            | "png" | "jpg" | "jpeg" | "gif" | "bmp"
            | "mp4" | "avi" | "wav" | "mp3"
            | "zip" | "tar" | "gz" | "bz2"
            | "so" | "dylib" | "dll"
    )
}
