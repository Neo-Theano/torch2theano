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

/// Convert an entire directory tree from PyTorch to Neo Theano.
pub fn convert_directory(
    input: &Path,
    output: &Path,
    dry_run: bool,
    verbose: bool,
) -> Result<Report> {
    let mut report = Report::new();

    for entry in WalkDir::new(input)
        .into_iter()
        .filter_entry(|e| !should_skip(e))
    {
        let entry = entry?;
        let src_path = entry.path();

        // Compute the relative path and the destination path.
        let rel = src_path
            .strip_prefix(input)
            .unwrap_or(src_path);
        let dst_path = output.join(rel);

        if entry.file_type().is_dir() {
            if !dry_run {
                fs::create_dir_all(&dst_path)
                    .with_context(|| format!("creating dir {}", dst_path.display()))?;
            }
            continue;
        }

        report.files_scanned += 1;
        let filename = src_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        // Decide how to handle this file.
        if is_python_file(src_path) {
            let source = fs::read_to_string(src_path)
                .with_context(|| format!("reading {}", src_path.display()))?;
            let result = transforms::transform_python(&source);

            let fr = FileReport {
                path: rel.display().to_string(),
                changes: result.changes,
            };

            if fr.change_count() > 0 || fr.warning_count() > 0 {
                report.files_transformed += 1;
                if verbose {
                    report::print_file_detail(&fr);
                }
            } else {
                report.files_copied += 1;
            }

            if !dry_run {
                if let Some(parent) = dst_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(&dst_path, &result.output)
                    .with_context(|| format!("writing {}", dst_path.display()))?;
            }

            report.file_reports.push(fr);
        } else if is_dep_file(&filename) {
            let source = fs::read_to_string(src_path)
                .with_context(|| format!("reading {}", src_path.display()))?;
            let result = transforms::transform_dependency_file(&source, &filename);

            let fr = FileReport {
                path: rel.display().to_string(),
                changes: result.changes,
            };

            if fr.change_count() > 0 || fr.warning_count() > 0 {
                report.files_transformed += 1;
                if verbose {
                    report::print_file_detail(&fr);
                }
            } else {
                report.files_copied += 1;
            }

            if !dry_run {
                if let Some(parent) = dst_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(&dst_path, &result.output)
                    .with_context(|| format!("writing {}", dst_path.display()))?;
            }

            report.file_reports.push(fr);
        } else if is_binary_or_large(src_path) {
            report.files_skipped += 1;
        } else {
            // Copy non-Python files as-is.
            report.files_copied += 1;
            if !dry_run {
                if let Some(parent) = dst_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::copy(src_path, &dst_path)
                    .with_context(|| {
                        format!("copying {} -> {}", src_path.display(), dst_path.display())
                    })?;
            }
        }
    }

    if !dry_run && report.files_transformed > 0 {
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
