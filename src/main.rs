mod converter;
mod report;
mod transforms;

use anyhow::{Context, Result};
use clap::Parser;
use colored::*;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "torch2theano")]
#[command(version)]
#[command(about = "Convert PyTorch Python source code to Neo Theano")]
#[command(long_about = "Walks a PyTorch repository, rewrites imports and API calls \
    from PyTorch to Neo Theano, copies the result to an output directory, and \
    prints a conversion report.")]
struct Cli {
    /// Path to the PyTorch repository to convert
    #[arg(value_name = "INPUT_DIR")]
    input: PathBuf,

    /// Output directory for converted code (default: <input>_theano)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Dry run: show what would change without writing files
    #[arg(long)]
    dry_run: bool,

    /// Show detailed per-file change descriptions
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    let input = cli
        .input
        .canonicalize()
        .with_context(|| format!("input path not found: {}", cli.input.display()))?;

    if !input.is_dir() {
        anyhow::bail!("input path is not a directory: {}", input.display());
    }

    let output = match cli.output {
        Some(p) => p,
        None => {
            let name = input
                .file_name()
                .map(|n| format!("{}_theano", n.to_string_lossy()))
                .unwrap_or_else(|| "output_theano".into());
            input.parent().unwrap_or(&input).join(name)
        }
    };

    println!(
        "{} {} -> {}",
        "torch2theano".bold().cyan(),
        input.display(),
        output.display()
    );

    if cli.dry_run {
        println!("{}", "(dry run)".yellow());
    }

    let report = converter::convert_directory(&input, &output, cli.dry_run, cli.verbose)?;
    report::print_report(&report);

    Ok(())
}
