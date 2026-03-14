use crate::transforms::{Change, ChangeKind};
use colored::*;
use std::collections::HashMap;

/// Aggregate conversion report across all files.
pub struct Report {
    pub files_scanned: usize,
    pub files_transformed: usize,
    pub files_copied: usize,
    pub files_skipped: usize,
    pub file_reports: Vec<FileReport>,
}

pub struct FileReport {
    pub path: String,
    pub changes: Vec<Change>,
}

impl Report {
    pub fn new() -> Self {
        Self {
            files_scanned: 0,
            files_transformed: 0,
            files_copied: 0,
            files_skipped: 0,
            file_reports: Vec::new(),
        }
    }

    pub fn total_changes(&self) -> usize {
        self.file_reports.iter().map(|f| f.change_count()).sum()
    }

    pub fn total_warnings(&self) -> usize {
        self.file_reports
            .iter()
            .map(|f| f.warning_count())
            .sum()
    }
}

impl FileReport {
    pub fn change_count(&self) -> usize {
        self.changes
            .iter()
            .filter(|c| c.kind != ChangeKind::Warning)
            .count()
    }

    pub fn warning_count(&self) -> usize {
        self.changes
            .iter()
            .filter(|c| c.kind == ChangeKind::Warning)
            .count()
    }
}

pub fn print_report(report: &Report) {
    println!();
    println!("{}", "─── Conversion Report ───".bold());
    println!(
        "  files scanned:     {}",
        report.files_scanned.to_string().cyan()
    );
    println!(
        "  files transformed: {}",
        report.files_transformed.to_string().green()
    );
    println!(
        "  files copied:      {}",
        report.files_copied.to_string().normal()
    );
    println!(
        "  files skipped:     {}",
        report.files_skipped.to_string().normal()
    );
    println!(
        "  total changes:     {}",
        report.total_changes().to_string().green()
    );

    let total_warnings = report.total_warnings();
    if total_warnings > 0 {
        println!(
            "  warnings:          {}",
            total_warnings.to_string().yellow()
        );
    }

    // ── Per-kind breakdown ────────────────────────────────────────
    let mut kind_counts: HashMap<&str, usize> = HashMap::new();
    for fr in &report.file_reports {
        for c in &fr.changes {
            if c.kind != ChangeKind::Warning {
                *kind_counts.entry(kind_label(c.kind)).or_default() += 1;
            }
        }
    }
    if !kind_counts.is_empty() {
        println!();
        println!("{}", "  Changes by category:".dimmed());
        for (label, count) in &kind_counts {
            println!("    {:<14} {}", label, count);
        }
    }

    // ── Warnings ─────────────────────────────────────────────────
    if total_warnings > 0 {
        println!();
        println!("{}", "  Warnings (require manual review):".yellow().bold());
        for fr in &report.file_reports {
            for c in &fr.changes {
                if c.kind == ChangeKind::Warning {
                    println!(
                        "    {}:{} {}",
                        fr.path.dimmed(),
                        c.line.to_string().dimmed(),
                        c.description.yellow()
                    );
                }
            }
        }
    }

    println!();
}

/// Verbose per-file detail (called from converter when --verbose).
pub fn print_file_detail(fr: &FileReport) {
    if fr.changes.is_empty() {
        return;
    }
    println!("  {}", fr.path.bold());
    for c in &fr.changes {
        let tag = match c.kind {
            ChangeKind::Import => "import".green(),
            ChangeKind::Namespace => "ns".blue(),
            ChangeKind::Dependency => "dep".magenta(),
            ChangeKind::Warning => "warn".yellow(),
        };
        println!("    L{:<5} [{}] {}", c.line, tag, c.description);
    }
}

fn kind_label(k: ChangeKind) -> &'static str {
    match k {
        ChangeKind::Import => "imports",
        ChangeKind::Namespace => "namespaces",
        ChangeKind::Dependency => "dependencies",
        ChangeKind::Warning => "warnings",
    }
}
