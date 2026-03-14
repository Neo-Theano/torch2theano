//! Rust code generator — transforms analyzed Python structures into Neo Theano Rust.
//!
//! Generates `Cargo.toml` and `src/main.rs` for a standalone Rust project
//! that uses the Neo Theano framework.

use regex::Regex;

use crate::analyzer::*;
use crate::transforms::{Change, ChangeKind};

/// Result of generating Rust code for a project.
pub struct CodegenResult {
    pub cargo_toml: String,
    pub main_rs: String,
    pub changes: Vec<Change>,
}

/// Generate a complete Rust project from analyzed Python files.
/// `project_name` is used for the Cargo.toml package name.
pub fn generate_project_named(files: &[(String, PyFile)], project_name: &str) -> CodegenResult {
    let mut changes = Vec::new();
    let mut all_imports = Vec::new();
    let mut all_classes = Vec::new();
    let mut all_functions = Vec::new();
    let mut all_top_level = Vec::new();

    for (filename, pyfile) in files {
        for imp in &pyfile.imports {
            all_imports.push((filename.clone(), imp.clone()));
        }
        for class in &pyfile.classes {
            all_classes.push((filename.clone(), class));
        }
        for func in &pyfile.functions {
            all_functions.push((filename.clone(), func));
        }
        for stmt in &pyfile.top_level {
            all_top_level.push((filename.clone(), stmt));
        }
    }

    let all_py_imports: Vec<PyImport> = all_imports.iter().map(|(_, i)| i.clone()).collect();
    let import_analysis = analyze_imports(&all_py_imports);

    for (filename, imp) in &all_imports {
        if imp.module.starts_with("torchvision")
            || imp.module.starts_with("torchaudio")
            || imp.module.starts_with("torchtext")
        {
            changes.push(Change {
                line: imp.line,
                kind: ChangeKind::Warning,
                description: format!(
                    "{} has no direct Neo Theano equivalent – manual porting required",
                    imp.module
                ),
                original: format!("{}:{}", filename, imp.line),
                replacement: String::new(),
            });
        }
        if imp.module.contains("torch.jit")
            || imp.module.contains("torch.onnx")
            || imp.module.contains("torch.distributed")
        {
            changes.push(Change {
                line: imp.line,
                kind: ChangeKind::Warning,
                description: format!("{} – review Neo Theano equivalent", imp.module),
                original: format!("{}:{}", filename, imp.line),
                replacement: String::new(),
            });
        }
    }

    let cargo_toml = generate_cargo_toml_named(files, project_name);
    changes.push(Change {
        line: 0,
        kind: ChangeKind::Dependency,
        description: "Generated Cargo.toml with theano dependency".to_string(),
        original: String::new(),
        replacement: "Cargo.toml".to_string(),
    });

    // Generate main.rs
    let main_rs = generate_main_rs(files, &import_analysis, &mut changes);

    CodegenResult {
        cargo_toml,
        main_rs,
        changes,
    }
}

/// Generate Cargo.toml for the output project.
fn generate_cargo_toml_named(files: &[(String, PyFile)], name: &str) -> String {

    // Check if argparse is used (need clap)
    let uses_argparse = files.iter().any(|(_, f)| {
        f.imports
            .iter()
            .any(|i| i.module == "argparse")
            || f.top_level
                .iter()
                .any(|s| s.text.contains("argparse"))
    });

    let mut deps = vec![
        r#"theano = { git = "https://github.com/Neo-Theano/theano.git", features = ["full"] }"#
            .to_string(),
    ];

    if uses_argparse {
        deps.push(r#"clap = { version = "4", features = ["derive"] }"#.to_string());
    }

    // Check for random usage
    let uses_random = files.iter().any(|(_, f)| {
        f.imports.iter().any(|i| i.module == "random")
    });
    if uses_random {
        deps.push(r#"rand = "0.8""#.to_string());
    }

    // Check for save/load usage (need theano-serialize)
    let uses_save_load = files.iter().any(|(_, f)| {
        f.top_level
            .iter()
            .any(|s| s.text.contains("torch.save") || s.text.contains("torch.load"))
    });
    if uses_save_load {
        deps.push(
            r#"theano-serialize = { git = "https://github.com/Neo-Theano/theano.git" }"#
                .to_string(),
        );
    }

    let deps_str = deps
        .iter()
        .map(|d| format!("{d}"))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"[package]
name = "{name}"
version = "0.1.0"
edition = "2021"

[dependencies]
{deps_str}
"#
    )
}

/// Generate the main.rs Rust source file.
fn generate_main_rs(
    files: &[(String, PyFile)],
    import_analysis: &ImportAnalysis,
    changes: &mut Vec<Change>,
) -> String {
    let mut out = String::new();

    // Header comment
    out.push_str("// Converted from PyTorch Python to Neo Theano Rust by torch2theano.\n");
    out.push_str("//\n");

    // Warnings header
    if !import_analysis.torchvision_imports.is_empty() {
        out.push_str("// Manual attention required:\n");
        for imp in &import_analysis.torchvision_imports {
            out.push_str(&format!(
                "//   - {} has no Neo Theano equivalent\n",
                imp.module
            ));
        }
        for imp in &import_analysis.unsupported_imports {
            out.push_str(&format!(
                "//   - {} has no Neo Theano equivalent\n",
                imp.module
            ));
        }
        out.push_str("//\n");
    }

    out.push('\n');

    // Use statements
    out.push_str("use theano::prelude::*;\n");

    if import_analysis.uses_nn {
        out.push_str("use theano::nn::*;\n");
        changes.push(Change {
            line: 0,
            kind: ChangeKind::Import,
            description: "torch.nn -> theano::nn".to_string(),
            original: String::new(),
            replacement: String::new(),
        });
    }
    if import_analysis.uses_optim {
        out.push_str("use theano::optim::{Adam, SGD, Optimizer};\n");
        changes.push(Change {
            line: 0,
            kind: ChangeKind::Import,
            description: "torch.optim -> theano::optim".to_string(),
            original: String::new(),
            replacement: String::new(),
        });
    }
    if import_analysis.uses_data {
        out.push_str("// use theano::data::{DataLoader, Dataset};\n");
    }

    out.push_str("use theano_types::Device;\n");

    // Check if save/load is used
    let uses_save_load = files.iter().any(|(_, f)| {
        f.top_level
            .iter()
            .any(|s| s.text.contains("torch.save") || s.text.contains("torch.load"))
    });
    if uses_save_load {
        out.push_str("use theano_serialize::{save_state_dict, load_state_dict};\n");
    }

    // Check if clap is needed
    let uses_argparse = files.iter().any(|(_, f)| {
        f.imports.iter().any(|i| i.module == "argparse")
    });
    if uses_argparse {
        out.push_str("use clap::Parser;\n");
    }

    let uses_random = files.iter().any(|(_, f)| {
        f.imports.iter().any(|i| i.module == "random")
    });
    if uses_random {
        out.push_str("use rand::Rng;\n");
    }

    out.push('\n');

    // Generate structs and impls for nn.Module classes
    for (filename, pyfile) in files {
        for class in &pyfile.classes {
            if is_nn_module(class) {
                out.push_str(&generate_module_struct(class, changes, filename));
                out.push('\n');
            }
        }
    }

    // Generate standalone functions
    for (_filename, pyfile) in files {
        for func in &pyfile.functions {
            out.push_str(&generate_function(func));
            out.push('\n');
        }
    }

    // Check for argparse usage and generate clap struct
    let argparse_args = extract_argparse_args(files);
    if !argparse_args.is_empty() {
        out.push_str("#[derive(Parser)]\n");
        out.push_str("#[command(about = \"Converted from PyTorch\")]\n");
        out.push_str("struct Args {\n");
        for arg in &argparse_args {
            if let Some(ref help) = arg.help {
                out.push_str(&format!("    /// {}\n", help));
            }
            if let Some(ref default) = arg.default {
                out.push_str(&format!(
                    "    #[arg(long, default_value_t = {})]\n",
                    rust_default_value(default, &arg.arg_type)
                ));
            } else if arg.is_flag {
                out.push_str("    #[arg(long)]\n");
            } else {
                out.push_str("    #[arg(long)]\n");
            }
            let rust_type = match arg.arg_type.as_str() {
                "int" => "usize",
                "float" => "f64",
                "str" | "" => "String",
                "bool" => "bool",
                _ => "String",
            };
            if arg.is_flag {
                out.push_str(&format!("    {}: bool,\n", arg.rust_name));
            } else if arg.default.is_none() {
                out.push_str(&format!("    {}: Option<{}>,\n", arg.rust_name, rust_type));
            } else {
                out.push_str(&format!("    {}: {},\n", arg.rust_name, rust_type));
            }
            out.push('\n');
        }
        out.push_str("}\n\n");
    }

    // Generate main function
    out.push_str("fn main() {\n");

    // Parse args if argparse was used
    if !argparse_args.is_empty() {
        out.push_str("    let args = Args::parse();\n\n");
    }

    // Collect top-level statements that look like they belong in main
    let skip_argparse = !argparse_args.is_empty();
    let mut in_argparse_block = false;

    // Track indentation depth to emit closing braces
    let mut indent_stack: Vec<usize> = vec![0]; // stack of Python indentation levels

    for (_filename, pyfile) in files {
        // Join multi-line statements at top level too
        let raw_stmts: Vec<String> = pyfile.top_level.iter().map(|s| s.text.clone()).collect();
        let joined = join_multiline_exprs(&raw_stmts);

        for stmt_text in &joined {
            let trimmed = stmt_text.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.starts_with('#') {
                // Keep meaningful comments
                let comment = trimmed.trim_start_matches('#').trim();
                if !comment.is_empty() {
                    let rust_indent = "    ".repeat(indent_stack.len());
                    out.push_str(&format!("{}// {}\n", rust_indent, comment));
                }
                continue;
            }
            // Skip import-like lines
            if trimmed.starts_with("import ") || trimmed.starts_with("from ") {
                continue;
            }

            // Skip argparse setup lines
            if skip_argparse {
                if trimmed.contains("argparse.ArgumentParser")
                    || trimmed.contains("parser.add_argument")
                    || trimmed.contains("parser.parse_args")
                {
                    in_argparse_block = true;
                    continue;
                }
                if in_argparse_block && (trimmed.starts_with("parser.") || trimmed.starts_with("opt = ")) {
                    continue;
                }
                in_argparse_block = false;
            }

            // Calculate Python indentation level
            let py_indent = stmt_text.len() - stmt_text.trim_start().len();

            // elif/else already include closing `}` in their translation,
            // so we pop the stack without emitting an extra brace.
            let is_elif_else = trimmed.starts_with("elif ") || trimmed == "else:";

            // Close blocks when indentation decreases
            while indent_stack.len() > 1 && py_indent <= *indent_stack.last().unwrap() {
                indent_stack.pop();
                if !is_elif_else {
                    let rust_indent = "    ".repeat(indent_stack.len());
                    out.push_str(&format!("{}}}\n", rust_indent));
                }
            }

            let rust_indent = "    ".repeat(indent_stack.len());
            let rust_line = translate_statement(trimmed);

            // If the translated line opens a block (ends with {), push indent level
            if rust_line.ends_with('{') {
                out.push_str(&format!("{}{}\n", rust_indent, rust_line));
                indent_stack.push(py_indent);
            } else if !rust_line.is_empty() {
                out.push_str(&format!("{}{}\n", rust_indent, rust_line));
            }
        }
    }

    // Close any remaining open blocks
    while indent_stack.len() > 1 {
        indent_stack.pop();
        let rust_indent = "    ".repeat(indent_stack.len());
        out.push_str(&format!("{}}}\n", rust_indent));
    }

    out.push_str("}\n");

    out
}

/// Generate a Rust struct + Module impl for a Python nn.Module class.
fn generate_module_struct(
    class: &PyClass,
    changes: &mut Vec<Change>,
    filename: &str,
) -> String {
    let mut out = String::new();

    out.push_str(&format!("/// {} neural network module.\n", class.name));
    out.push_str(&format!("struct {} {{\n", class.name));

    // Parse __init__ to find layer fields
    let fields = if let Some(init) = get_init_method(class) {
        extract_layer_fields(init)
    } else {
        Vec::new()
    };

    for (name, layer_type) in &fields {
        out.push_str(&format!("    {}: {},\n", name, layer_type));
    }

    out.push_str("}\n\n");

    // Constructor impl
    out.push_str(&format!("impl {} {{\n", class.name));

    if let Some(init) = get_init_method(class) {
        // Extract constructor args (skip self)
        let args: Vec<&str> = init
            .args
            .iter()
            .filter(|a| a.as_str() != "self")
            .map(|a| a.as_str())
            .collect();

        let arg_list: String = args
            .iter()
            .map(|a| format!("{}: usize", a))
            .collect::<Vec<_>>()
            .join(", ");

        out.push_str(&format!("    fn new({}) -> Self {{\n", arg_list));

        // Generate field initializations from __init__ body
        let field_inits = generate_field_inits(init, &fields);
        for init_line in &field_inits {
            out.push_str(&format!("        {}\n", init_line));
        }

        // Self constructor
        out.push_str("        Self {\n");
        for (name, _) in &fields {
            out.push_str(&format!("            {},\n", name));
        }
        out.push_str("        }\n");
        out.push_str("    }\n");
    }

    out.push_str("}\n\n");

    // Module trait impl
    out.push_str(&format!("impl Module for {} {{\n", class.name));

    // forward method
    if let Some(forward) = get_forward_method(class) {
        out.push_str("    fn forward(&self, input: &Variable) -> Variable {\n");
        let forward_body = translate_forward_body(forward, &fields);
        for line in &forward_body {
            out.push_str(&format!("        {}\n", line));
        }
        out.push_str("    }\n\n");
    } else {
        out.push_str("    fn forward(&self, input: &Variable) -> Variable {\n");
        out.push_str("        // TODO: implement forward pass\n");
        out.push_str("        input.clone()\n");
        out.push_str("    }\n\n");
    }

    // parameters method
    out.push_str("    fn parameters(&self) -> Vec<Variable> {\n");
    if fields.len() == 1 {
        out.push_str(&format!(
            "        self.{}.parameters()\n",
            fields[0].0
        ));
    } else if !fields.is_empty() {
        out.push_str(&format!(
            "        let mut params = self.{}.parameters();\n",
            fields[0].0
        ));
        for (name, _) in fields.iter().skip(1) {
            out.push_str(&format!(
                "        params.extend(self.{}.parameters());\n",
                name
            ));
        }
        out.push_str("        params\n");
    } else {
        out.push_str("        Vec::new()\n");
    }
    out.push_str("    }\n");

    out.push_str("}\n");

    changes.push(Change {
        line: class.line,
        kind: ChangeKind::Namespace,
        description: format!(
            "class {}(nn.Module) -> struct {} + impl Module",
            class.name, class.name
        ),
        original: format!("{}:{}", filename, class.line),
        replacement: String::new(),
    });

    out
}

/// Extract field names and their Rust types from __init__ body.
/// Handles multi-line expressions by joining continuation lines.
fn extract_layer_fields(init: &PyMethod) -> Vec<(String, String)> {
    let re_self_assign = Regex::new(r"self\.(\w+)\s*=\s*(.+)").unwrap();
    let mut fields = Vec::new();

    // First, join multi-line expressions (track open parens)
    let joined_lines = join_multiline_exprs(&init.body);

    for line in &joined_lines {
        let trimmed = line.trim();

        // Skip super().__init__()
        if trimmed.contains("super()") || trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }

        if let Some(caps) = re_self_assign.captures(trimmed) {
            let name = caps[1].to_string();
            let value = caps[2].trim().to_string();

            // Skip simple scalar assignments like self.ngpu = ngpu
            if is_layer_construction(&value) {
                let rust_type = infer_rust_type(&value);
                fields.push((name, rust_type));
            }
        }
    }

    fields
}

/// Join lines that are continuations of multi-line expressions.
/// Detects unclosed parentheses and concatenates until balanced.
/// Strips Python comments from each line before joining to avoid
/// `#` characters in comments breaking the expression parsing.
fn join_multiline_exprs(lines: &[String]) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut current_prefix = String::new(); // preserved leading whitespace of first line
    let mut paren_depth: i32 = 0;

    for line in lines {
        let trimmed = line.trim();

        // Strip inline comments from this line before joining.
        // This prevents # in comments from breaking downstream parsing.
        let clean = strip_line_comment(trimmed);
        let clean = clean.trim();

        if clean.is_empty() && paren_depth == 0 {
            // Standalone empty/comment line outside an expression — preserve indentation
            result.push(line.to_string());
            continue;
        }

        if current.is_empty() {
            // Preserve the original leading whitespace of the first line
            let indent_len = line.len() - line.trim_start().len();
            current_prefix = line[..indent_len].to_string();
            current = clean.to_string();
        } else if !clean.is_empty() {
            current.push(' ');
            current.push_str(clean);
        }

        // Count parens in the cleaned line
        for ch in clean.chars() {
            match ch {
                '(' => paren_depth += 1,
                ')' => paren_depth -= 1,
                _ => {}
            }
        }

        if paren_depth <= 0 {
            // Re-attach original indentation so callers can track indent level
            result.push(format!("{}{}", current_prefix, current));
            current.clear();
            current_prefix.clear();
            paren_depth = 0;
        }
    }

    if !current.is_empty() {
        result.push(format!("{}{}", current_prefix, current));
    }

    result
}

/// Strip a Python comment from a single line, respecting quotes.
fn strip_line_comment(line: &str) -> &str {
    let mut in_single = false;
    let mut in_double = false;
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let ch = bytes[i] as char;
        match ch {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '#' if !in_single && !in_double => {
                return line[..i].trim_end();
            }
            '\\' => {
                i += 1; // skip escaped char
            }
            _ => {}
        }
        i += 1;
    }
    line
}

/// Check if a value looks like an nn layer construction.
fn is_layer_construction(value: &str) -> bool {
    value.starts_with("nn.")
        || value.starts_with("torch.nn.")
        || value.contains("Sequential")
        || value.contains("Linear")
        || value.contains("Conv2d")
        || value.contains("Conv1d")
        || value.contains("ConvTranspose2d")
        || value.contains("BatchNorm")
        || value.contains("Dropout")
        || value.contains("Embedding")
        || value.contains("MaxPool")
        || value.contains("AvgPool")
        || value.contains("AdaptiveAvgPool")
}

/// Infer a Rust type from a Python layer construction expression.
fn infer_rust_type(value: &str) -> String {
    if value.contains("Sequential") {
        "Sequential".to_string()
    } else if value.contains("ConvTranspose2d") {
        "Conv2d".to_string() // TODO: ConvTranspose2d when available
    } else if value.contains("Conv2d") {
        "Conv2d".to_string()
    } else if value.contains("Conv1d") {
        "Conv1d".to_string()
    } else if value.contains("Linear") {
        "Linear".to_string()
    } else if value.contains("BatchNorm2d") || value.contains("BatchNorm1d") {
        "BatchNorm1d".to_string()
    } else if value.contains("Dropout") {
        "Dropout".to_string()
    } else if value.contains("MaxPool2d") {
        "MaxPool2d".to_string()
    } else if value.contains("AdaptiveAvgPool2d") {
        "AdaptiveAvgPool2d".to_string()
    } else if value.contains("AvgPool2d") {
        "AvgPool2d".to_string()
    } else if value.contains("Embedding") {
        "Embedding".to_string()
    } else if value.contains("ReLU") {
        "ReLU".to_string()
    } else if value.contains("Sigmoid") {
        "Sigmoid".to_string()
    } else if value.contains("Tanh") {
        "Tanh".to_string()
    } else {
        "/* TODO: unknown layer */".to_string()
    }
}

/// Generate field initialization lines from __init__ body.
fn generate_field_inits(init: &PyMethod, fields: &[(String, String)]) -> Vec<String> {
    let mut inits = Vec::new();
    let re_self_assign = Regex::new(r"self\.(\w+)\s*=\s*(.+)").unwrap();

    let joined_lines = join_multiline_exprs(&init.body);

    for line in &joined_lines {
        let trimmed = line.trim();
        if trimmed.contains("super()") || trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }

        if let Some(caps) = re_self_assign.captures(trimmed) {
            let name = &caps[1];
            let value = caps[2].trim();

            // Only generate inits for fields we extracted
            if fields.iter().any(|(n, _)| n == name) {
                let rust_expr = translate_layer_construction(value);
                inits.push(format!("let {} = {};", name, rust_expr));
            }
        }
    }

    inits
}

/// Translate a Python nn layer construction to Rust.
fn translate_layer_construction(py: &str) -> String {
    let trimmed = py.trim().trim_end_matches(',');

    // nn.Sequential(...) — multi-line, needs special handling
    if trimmed.contains("Sequential") {
        return translate_sequential(trimmed);
    }

    // nn.Linear(in, out)
    let re_linear = Regex::new(r"nn\.Linear\(\s*(\w+)\s*,\s*(\w+)\s*\)").unwrap();
    if let Some(caps) = re_linear.captures(trimmed) {
        return format!("Linear::new({}, {})", &caps[1], &caps[2]);
    }

    // nn.Conv2d / nn.ConvTranspose2d with various argument patterns
    // Handles expressions like ngf * 8 as arguments
    let re_conv = Regex::new(
        r"nn\.(Conv(?:Transpose)?2d)\(\s*(.+?)\s*,\s*(.+?)\s*,\s*(\d+)\s*(?:,\s*(\d+))?\s*(?:,\s*(\d+))?\s*(?:,\s*bias\s*=\s*(True|False))?\s*\)",
    ).unwrap();
    if let Some(caps) = re_conv.captures(trimmed) {
        let _layer_type = &caps[1]; // Conv2d or ConvTranspose2d
        let in_ch = caps[2].trim();
        let out_ch = caps[3].trim();
        let kernel = &caps[4];
        let stride = caps.get(5).map_or("1", |m| m.as_str());
        let padding = caps.get(6).map_or("0", |m| m.as_str());
        let bias = caps
            .get(7)
            .map_or("true", |m| if m.as_str() == "False" { "false" } else { "true" });
        // TODO: ConvTranspose2d needs its own Rust type when available
        return format!(
            "Conv2d::with_options({}, {}, ({k}, {k}), ({s}, {s}), ({p}, {p}), {bias})",
            in_ch,
            out_ch,
            k = kernel,
            s = stride,
            p = padding,
            bias = bias
        );
    }

    // nn.BatchNorm2d(features) / nn.BatchNorm1d(features)
    // Handle expressions like ngf * 8
    let re_bn = Regex::new(r"nn\.BatchNorm[12]d\(\s*(.+?)\s*\)").unwrap();
    if let Some(caps) = re_bn.captures(trimmed) {
        return format!("BatchNorm1d::new({})", caps[1].trim());
    }

    // nn.LeakyReLU(negative_slope, inplace=...)
    let re_leaky = Regex::new(r"nn\.LeakyReLU\(\s*([\d.]+)").unwrap();
    if let Some(caps) = re_leaky.captures(trimmed) {
        // TODO: LeakyReLU not yet in Neo Theano — use ReLU as placeholder
        return format!("ReLU /* TODO: LeakyReLU({}) */", &caps[1]);
    }

    // nn.Dropout(p)
    let re_dropout = Regex::new(r"nn\.Dropout\(\s*([\d.]+)\s*\)").unwrap();
    if let Some(caps) = re_dropout.captures(trimmed) {
        return format!("Dropout::new({})", &caps[1]);
    }

    // nn.MaxPool2d(kernel)
    let re_maxpool = Regex::new(r"nn\.MaxPool2d\(\s*(\d+)\s*\)").unwrap();
    if let Some(caps) = re_maxpool.captures(trimmed) {
        return format!("MaxPool2d::new({})", &caps[1]);
    }

    // nn.Embedding(num, dim)
    let re_embed = Regex::new(r"nn\.Embedding\(\s*(\S+?)\s*,\s*(\S+?)\s*\)").unwrap();
    if let Some(caps) = re_embed.captures(trimmed) {
        return format!("Embedding::new({}, {})", &caps[1], &caps[2]);
    }

    // Simple activations
    if trimmed.contains("nn.ReLU") {
        return "ReLU".to_string();
    }
    if trimmed.contains("nn.Sigmoid") {
        return "Sigmoid".to_string();
    }
    if trimmed.contains("nn.Tanh") {
        return "Tanh".to_string();
    }

    // Fallback: comment the original
    format!("/* TODO: translate {} */", trimmed)
}

/// Translate an nn.Sequential(...) block.
fn translate_sequential(py: &str) -> String {
    // Extract contents between nn.Sequential( and the matching closing )
    // Must handle nested parentheses from layer constructors like nn.Conv2d(...)
    if let Some(start) = py.find("nn.Sequential(") {
        let content_start = start + "nn.Sequential(".len();
        let bytes = py.as_bytes();
        let mut depth = 1i32;
        let mut end = content_start;
        while end < bytes.len() && depth > 0 {
            match bytes[end] as char {
                '(' => depth += 1,
                ')' => depth -= 1,
                _ => {}
            }
            if depth > 0 {
                end += 1;
            }
        }

        let inner = &py[content_start..end];
        let inner = inner.trim();

        if inner.is_empty() {
            return "Sequential::new(vec![])".to_string();
        }

        // Parse individual layer expressions
        let layers = split_sequential_args(inner);
        let mut out = "Sequential::new(vec![])\n".to_string();
        for layer in &layers {
            let trimmed = layer.trim().trim_matches(',');
            if trimmed.is_empty() {
                continue;
            }
            let rust_layer = translate_layer_construction(trimmed);
            out.push_str(&format!("            .add({})\n", rust_layer));
        }
        return out.trim_end().to_string();
    }

    "Sequential::new(vec![])".to_string()
}

/// Split nn.Sequential arguments, respecting nested parentheses.
/// Strips out Python comments (# ...) that appear between layers.
fn split_sequential_args(inner: &str) -> Vec<String> {
    // First, strip Python comments from the inner content
    let stripped = strip_inline_comments(inner);

    let mut args = Vec::new();
    let mut current = String::new();
    let mut depth = 0;

    for ch in stripped.chars() {
        match ch {
            '(' => {
                depth += 1;
                current.push(ch);
            }
            ')' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    args.push(trimmed);
                }
                current.clear();
            }
            '\n' | '\r' => {
                current.push(' ');
            }
            _ => current.push(ch),
        }
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        args.push(trimmed);
    }

    args
}

/// Strip Python comments (# ...) from a string, handling # inside strings.
fn strip_inline_comments(s: &str) -> String {
    let mut result = String::new();
    let mut in_single = false;
    let mut in_double = false;

    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        match ch {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '#' if !in_single && !in_double => {
                // Skip until end of line
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
                continue;
            }
            _ => {}
        }
        result.push(ch);
        i += 1;
    }

    result
}

/// Translate the body of a forward() method to Rust.
fn translate_forward_body(method: &PyMethod, fields: &[(String, String)]) -> Vec<String> {
    let mut lines = Vec::new();

    // Get input parameter name
    let input_name = method
        .args
        .iter()
        .find(|a| a.as_str() != "self")
        .map(|a| a.as_str())
        .unwrap_or("input");

    // Join multi-line expressions in the forward body
    let joined = join_multiline_exprs(&method.body);

    // Filter to just code lines (no comments, no empty)
    let code_lines: Vec<&str> = joined.iter()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();

    // Detect the common pattern: sequential layer calls + return
    // Look for the simplest interpretation of the forward body
    let has_self_main = code_lines.iter().any(|l| l.contains("self.main("));
    let has_data_parallel = code_lines.iter().any(|l| l.contains("data_parallel"));

    // If it's a simple "output = self.main(input); return output" or
    // has data_parallel branching, simplify to just the main call
    if has_self_main && has_data_parallel {
        // Simplified: just call self.main.forward(input)
        let has_view = code_lines.iter().any(|l| l.contains(".view("));
        if has_view {
            // Discriminator pattern: self.main(input).view(-1,1).squeeze(1)
            // Find the view/squeeze expression
            for line in &code_lines {
                if line.starts_with("return ") && line.contains(".view(") {
                    let rest = &line[7..]; // strip "return "
                    // Translate the chained view/squeeze
                    let translated = rest
                        .replace("output", "x")
                        .replace(".squeeze(1)", "");
                    let re_view = Regex::new(r"(\w+)\.view\(([^)]+)\)").unwrap();
                    if let Some(caps) = re_view.captures(&translated) {
                        let dims = &caps[2];
                        lines.push("let x = self.main.forward(input);".to_string());
                        lines.push(format!("x.reshape(&[{}]).unwrap()", dims));
                    } else {
                        lines.push("let x = self.main.forward(input);".to_string());
                        lines.push("// TODO: reshape output".to_string());
                        lines.push("x".to_string());
                    }
                    return lines;
                }
            }
            lines.push("let x = self.main.forward(input);".to_string());
            lines.push("x".to_string());
        } else {
            lines.push("self.main.forward(input)".to_string());
        }
        return lines;
    }

    // Check for a simple single-expression forward
    if code_lines.len() <= 2 {
        for py_line in &code_lines {
            let adjusted = py_line.replace(input_name, "input");
            let rust_line = translate_forward_statement(&adjusted, fields);

            // For return statements, strip "let output = " wrapper
            if adjusted.starts_with("return ") {
                let expr = translate_expression(&adjusted[7..], fields);
                lines.push(expr);
            } else {
                lines.push(rust_line);
            }
        }
        return lines;
    }

    // General case: translate line by line
    for py_line in &joined {
        let trimmed = py_line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('#') {
            lines.push(format!("// {}", trimmed.trim_start_matches('#').trim()));
            continue;
        }

        let adjusted = trimmed.replace(input_name, "input");
        let rust_line = translate_forward_statement(&adjusted, fields);
        lines.push(rust_line);
    }

    lines
}

/// Translate a single statement inside a forward() method.
fn translate_forward_statement(py: &str, fields: &[(String, String)]) -> String {
    let trimmed = py.trim();

    // return expr
    if let Some(rest) = trimmed.strip_prefix("return ") {
        let expr = translate_expression(rest.trim(), fields);
        return expr;
    }

    // x = self.layer(y)
    let re_assign = Regex::new(r"^(\w+)\s*=\s*(.+)$").unwrap();
    if let Some(caps) = re_assign.captures(trimmed) {
        let var = &caps[1];
        let expr = translate_expression(caps[2].trim(), fields);
        return format!("let {} = {};", var, expr);
    }

    // Fallback
    format!("// TODO: {}", trimmed)
}

/// Translate a Python expression to Rust, handling common patterns.
fn translate_expression(py: &str, fields: &[(String, String)]) -> String {
    let trimmed = py.trim();

    // self.layer(x) -> self.layer.forward(&x)
    let re_self_call = Regex::new(r"^self\.(\w+)\((\w+)\)$").unwrap();
    if let Some(caps) = re_self_call.captures(trimmed) {
        let field = &caps[1];
        let arg = &caps[2];
        if fields.iter().any(|(n, _)| n == field) {
            return format!("self.{}.forward(&{})", field, arg);
        }
    }

    // self.main(input) -> self.main.forward(input)
    let re_self_call2 = Regex::new(r"^self\.(\w+)\((.+)\)$").unwrap();
    if let Some(caps) = re_self_call2.captures(trimmed) {
        let field = &caps[1];
        let args = &caps[2];
        if fields.iter().any(|(n, _)| n == field) {
            let rust_args = translate_args(args);
            return format!("self.{}.forward({})", field, rust_args);
        }
    }

    // output.view(-1, 1).squeeze(1) pattern — must come before plain .view()
    if trimmed.contains(".view(") && trimmed.contains(".squeeze(") {
        let re = Regex::new(r"(\w+)\.view\(([^)]+)\)\.squeeze\((\d+)\)").unwrap();
        if let Some(caps) = re.captures(trimmed) {
            let var = &caps[1];
            let shape = &caps[2];
            return format!(
                "{}.reshape(&[{}]).unwrap()",
                var,
                shape
            );
        }
    }

    // x.view(-1, N) -> x.reshape(&[-1, N]).unwrap()
    let re_view = Regex::new(r"^(\w+)\.view\((.+)\)$").unwrap();
    if let Some(caps) = re_view.captures(trimmed) {
        let var = &caps[1];
        let dims = &caps[2];
        return format!("{}.reshape(&[{}]).unwrap()", var, dims);
    }

    // x.relu() / x.sigmoid() / x.tanh()
    for act in &["relu", "sigmoid", "tanh"] {
        let pat = format!(r"^(\w+)\.{}(?:\(\))?$", act);
        let re = Regex::new(&pat).unwrap();
        if let Some(caps) = re.captures(trimmed) {
            return format!("{}.{}().unwrap()", &caps[1], act);
        }
    }

    // torch.randn(...) -> Tensor::randn(&[...])
    let re_randn = Regex::new(r"^torch\.randn\((.+)\)$").unwrap();
    if let Some(caps) = re_randn.captures(trimmed) {
        let args = &caps[1];
        // Strip device= kwarg
        let clean = remove_kwargs(args);
        return format!("Variable::new(Tensor::randn(&[{}]))", clean);
    }

    // torch.zeros(...) -> Tensor::zeros(&[...])
    let re_zeros = Regex::new(r"^torch\.zeros\((.+)\)$").unwrap();
    if let Some(caps) = re_zeros.captures(trimmed) {
        let args = &caps[1];
        let clean = remove_kwargs(args);
        return format!("Variable::new(Tensor::zeros(&[{}]))", clean);
    }

    // torch.ones(...) -> Tensor::ones(&[...])
    let re_ones = Regex::new(r"^torch\.ones\((.+)\)$").unwrap();
    if let Some(caps) = re_ones.captures(trimmed) {
        let args = &caps[1];
        let clean = remove_kwargs(args);
        return format!("Variable::new(Tensor::ones(&[{}]))", clean);
    }

    // torch.full((shape,), val) -> Tensor::full(&[shape], val)
    let re_full = Regex::new(r"^torch\.full\(\((.+?)\),\s*(.+)\)$").unwrap();
    if let Some(caps) = re_full.captures(trimmed) {
        let shape = &caps[1];
        let val = remove_kwargs(&caps[2]);
        return format!("Variable::new(Tensor::full(&[{}], {}))", shape, val.trim());
    }

    // Fallback: return as comment
    format!("/* TODO: {} */", trimmed)
}

/// Translate function arguments, adding & for variable references.
fn translate_args(args: &str) -> String {
    let parts: Vec<&str> = args.split(',').collect();
    let translated: Vec<String> = parts
        .iter()
        .map(|a| {
            let trimmed = a.trim();
            if trimmed.starts_with('&') || trimmed.starts_with('"') || trimmed.parse::<f64>().is_ok()
            {
                trimmed.to_string()
            } else {
                format!("&{}", trimmed)
            }
        })
        .collect();
    translated.join(", ")
}

/// Remove keyword arguments from a Python call (device=..., dtype=..., etc.)
fn remove_kwargs(args: &str) -> String {
    let parts: Vec<&str> = args.split(',').collect();
    let positional: Vec<&str> = parts
        .iter()
        .filter(|a| !a.contains('='))
        .copied()
        .collect();
    positional.join(",")
}

/// Generate a Rust function from a Python function.
fn generate_function(func: &PyFunction) -> String {
    let mut out = String::new();

    let args: Vec<&str> = func
        .args
        .iter()
        .map(|a| a.as_str())
        .collect();

    // Simple function signature
    let arg_list = if args.is_empty() {
        String::new()
    } else {
        args.iter()
            .map(|a| format!("{}: /* TODO */", a))
            .collect::<Vec<_>>()
            .join(", ")
    };

    out.push_str(&format!("fn {}({}) {{\n", func.name, arg_list));

    for line in &func.body {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            out.push('\n');
            continue;
        }
        if trimmed.starts_with('#') {
            out.push_str(&format!("    // {}\n", trimmed.trim_start_matches('#').trim()));
        } else {
            out.push_str(&format!("    // TODO: {}\n", trimmed));
        }
    }

    out.push_str("}\n");
    out
}

/// Extracted argparse argument.
struct ArgparseArg {
    rust_name: String,
    arg_type: String,
    default: Option<String>,
    help: Option<String>,
    is_flag: bool,
}

/// Extract argparse arguments from top-level statements.
fn extract_argparse_args(files: &[(String, PyFile)]) -> Vec<ArgparseArg> {
    let re_add_arg = Regex::new(
        r"parser\.add_argument\(\s*'--([^']+)'",
    ).unwrap();
    let re_type = Regex::new(r"type\s*=\s*(\w+)").unwrap();
    let re_default_num = Regex::new(r"default\s*=\s*([\d.]+)").unwrap();
    let re_default_str = Regex::new(r"default\s*=\s*'([^']*)'").unwrap();
    let re_default_empty_str = Regex::new(r#"default\s*=\s*''"#).unwrap();
    let re_help = Regex::new(r#"help\s*=\s*['"]((?:[^'"\\]|\\.)*)['"]"#).unwrap();
    let re_action_store_true = Regex::new(r"action\s*=\s*'store_true'").unwrap();

    let mut args = Vec::new();

    for (_filename, pyfile) in files {
        let raw_stmts: Vec<String> = pyfile.top_level.iter().map(|s| s.text.clone()).collect();
        let joined = join_multiline_exprs(&raw_stmts);

        for stmt in &joined {
            let trimmed = stmt.trim();
            if let Some(caps) = re_add_arg.captures(trimmed) {
                let py_name = &caps[1];
                let rust_name = py_name.replace('-', "_").to_lowercase();

                let arg_type = re_type
                    .captures(trimmed)
                    .map(|c| c[1].to_string())
                    .unwrap_or_default();

                let is_flag = re_action_store_true.is_match(trimmed);

                let default = if is_flag {
                    Some("false".to_string())
                } else if let Some(dcaps) = re_default_num.captures(trimmed) {
                    Some(dcaps[1].to_string())
                } else if re_default_empty_str.is_match(trimmed) {
                    Some("String::new()".to_string())
                } else if let Some(dcaps) = re_default_str.captures(trimmed) {
                    Some(format!("\"{}\"", &dcaps[1]))
                } else {
                    None
                };

                let help = re_help
                    .captures(trimmed)
                    .map(|c| c[1].to_string());

                args.push(ArgparseArg {
                    rust_name,
                    arg_type,
                    default,
                    help,
                    is_flag,
                });
            }
        }
    }

    args
}

/// Convert a Python default value to Rust syntax.
fn rust_default_value(val: &str, arg_type: &str) -> String {
    if val == "false" || val == "true" {
        return val.to_string();
    }
    if val.starts_with('"') || val.starts_with("String::") {
        // For String types, use into()
        return format!("{}.into()", val);
    }
    match arg_type {
        "int" => val.to_string(),
        "float" => {
            // Ensure it has a decimal point for f64
            if val.contains('.') {
                val.to_string()
            } else {
                format!("{}.0", val)
            }
        }
        _ => val.to_string(),
    }
}

/// Translate a top-level Python statement to Rust (best-effort).
fn translate_statement(py: &str) -> String {
    let trimmed = py.trim();

    // Skip common Python boilerplate
    if trimmed.starts_with("from __future__") || trimmed == "pass" {
        return String::new();
    }

    // print(...) -> println!(...)
    let re_print = Regex::new(r"^print\((.+)\)$").unwrap();
    if let Some(caps) = re_print.captures(trimmed) {
        let content = &caps[1];
        // Handle f-strings
        if content.starts_with("f\"") || content.starts_with("f'") {
            return format!("// TODO: println!({});", content);
        }
        // Handle Python % format strings: 'format' % (args) or "format" % (args)
        let re_pct_fmt = Regex::new(r#"^['"](.+?)['"]\s*%\s*\((.+)\)$"#).unwrap();
        if let Some(fcaps) = re_pct_fmt.captures(content) {
            let fmt_str = &fcaps[1];
            let args = &fcaps[2];
            // Convert Python format specifiers to Rust: %d -> {}, %.4f -> {:.4}, %s -> {}
            let rust_fmt = fmt_str
                .replace("%d", "{}")
                .replace("%s", "{}")
                .replace("%.4f", "{:.4}")
                .replace("%.2f", "{:.2}")
                .replace("%.6f", "{:.6}")
                .replace("%f", "{}");
            return format!("println!(\"{}\", {});", rust_fmt, args);
        }
        let content = python_strings_to_rust(content);
        return format!("println!(\"{{}}\", {});", content);
    }

    // model = ModelName(...).to(device) -> let model = ModelName::new(...)
    let re_model_to = Regex::new(r"^(\w+)\s*=\s*(\w+)\(([^)]*)\)\.to\(\w+\)$").unwrap();
    if let Some(caps) = re_model_to.captures(trimmed) {
        return format!(
            "let {} = {}::new({});",
            &caps[1], &caps[2], &caps[3]
        );
    }

    // model = ModelName(...) for known nn.Module classes
    let re_model_new = Regex::new(r"^(\w+)\s*=\s*([A-Z]\w+)\(([^)]*)\)$").unwrap();
    if let Some(caps) = re_model_new.captures(trimmed) {
        let var = &caps[1];
        let class_name = &caps[2];
        let args = &caps[3];
        // Check if this looks like a class instantiation (capitalized name)
        if class_name.chars().next().map_or(false, |c| c.is_uppercase()) {
            return format!("let {} = {}::new({});", var, class_name, args);
        }
    }

    // model.apply(func) -> skip (Rust doesn't have apply)
    if trimmed.contains(".apply(") {
        return format!("// TODO: {}", trimmed);
    }

    // .load_state_dict(torch.load(path))
    let re_load_sd = Regex::new(r"(\w+)\.load_state_dict\(torch\.load\((.+?)\)\)").unwrap();
    if let Some(caps) = re_load_sd.captures(trimmed) {
        let model = &caps[1];
        let path = &caps[2];
        return format!(
            "// Load state dict for {model}\n    \
             let _bytes = std::fs::read({path}).expect(\"failed to read checkpoint\");\n    \
             let _state = theano_serialize::load_state_dict(&_bytes).expect(\"failed to load state dict\");\n    \
             // TODO: apply _state to {model}",
        );
    }
    if trimmed.contains("load_state_dict") {
        return format!("// TODO: {}", trimmed);
    }

    // torch.save(model.state_dict(), path)
    let re_save = Regex::new(r"torch\.save\((\w+)\.state_dict\(\),\s*(.+)\)").unwrap();
    if let Some(caps) = re_save.captures(trimmed) {
        let model = &caps[1];
        let path_expr = &caps[2];
        let rust_path = translate_python_format_string(path_expr);
        return format!(
            "let _bytes = theano_serialize::save_state_dict(&{model}.state_dict());\n    \
             std::fs::write({rust_path}, _bytes).expect(\"failed to save checkpoint\");",
        );
    }
    if trimmed.contains("torch.save") {
        return format!("// TODO: {}", trimmed);
    }

    // for epoch in range(niter):
    let re_for_range = Regex::new(r"^for\s+(\w+)\s+in\s+range\((.+)\):$").unwrap();
    if let Some(caps) = re_for_range.captures(trimmed) {
        return format!("for {} in 0..{} {{", &caps[1], &caps[2]);
    }

    // for i, data in enumerate(dataloader, 0):
    let re_for_enum = Regex::new(r"^for\s+(\w+),\s*(\w+)\s+in\s+enumerate\((\w+).*\):$").unwrap();
    if let Some(caps) = re_for_enum.captures(trimmed) {
        return format!("for ({}, {}) in {}.iter().enumerate() {{", &caps[1], &caps[2], &caps[3]);
    }

    // if condition:
    let re_if = Regex::new(r"^if\s+(.+):$").unwrap();
    if let Some(caps) = re_if.captures(trimmed) {
        let cond = translate_condition(&caps[1]);
        return format!("if {} {{", cond);
    }

    // elif condition:
    let re_elif = Regex::new(r"^elif\s+(.+):$").unwrap();
    if let Some(caps) = re_elif.captures(trimmed) {
        let cond = translate_condition(&caps[1]);
        return format!("}} else if {} {{", cond);
    }

    // else:
    if trimmed == "else:" {
        return "} else {".to_string();
    }

    // break
    if trimmed == "break" {
        return "break;".to_string();
    }

    // assert
    if trimmed.starts_with("assert ") {
        let rest = &trimmed[7..];
        return format!("assert!({});", rest);
    }

    // try: / except: / raise
    if trimmed == "try:" || trimmed.starts_with("except ") || trimmed.starts_with("raise ") {
        return format!("// TODO: {}", trimmed);
    }

    // Generic assignment
    let re_assign = Regex::new(r"^(\w+)\s*=\s*(.+)$").unwrap();
    if let Some(caps) = re_assign.captures(trimmed) {
        let var = &caps[1];
        let val = caps[2].trim();

        // criterion = nn.BCELoss()
        if val.contains("nn.BCELoss") {
            return format!("let {} = BCELoss::new();", var);
        }
        if val.contains("nn.MSELoss") {
            return format!("let {} = MSELoss::new();", var);
        }
        if val.contains("nn.CrossEntropyLoss") {
            return format!("let {} = CrossEntropyLoss::new();", var);
        }

        // optim.Adam(...)
        let re_optim = Regex::new(r"optim\.(Adam|SGD)\((.+?)\.parameters\(\),\s*(?:lr\s*=\s*)?(\S+?)(?:,\s*betas\s*=\s*\((\S+?),\s*(\S+?)\))?\s*\)").unwrap();
        if let Some(ocaps) = re_optim.captures(val) {
            let optim_name = &ocaps[1];
            let model = &ocaps[2];
            let lr = &ocaps[3];
            let betas = ocaps.get(4).and_then(|b1| {
                ocaps.get(5).map(|b2| (b1.as_str(), b2.as_str()))
            });
            let mut expr = format!(
                "let mut {} = {}::new({}.parameters(), {})",
                var, optim_name, model, lr
            );
            if let Some((b1, b2)) = betas {
                expr.push_str(&format!(".betas({}, {})", b1, b2));
            }
            expr.push(';');
            return expr;
        }

        // torch.randn(...)
        if val.contains("torch.randn") {
            let re_randn = Regex::new(r"torch\.randn\((.+)\)").unwrap();
            if let Some(rcaps) = re_randn.captures(val) {
                let args = remove_kwargs(&rcaps[1]);
                return format!(
                    "let {} = Variable::new(Tensor::randn(&[{}]));",
                    var, args
                );
            }
        }

        // torch.full(...)
        if val.contains("torch.full") {
            let re_full = Regex::new(r"torch\.full\(\((.+?)\),\s*(.+)\)").unwrap();
            if let Some(fcaps) = re_full.captures(val) {
                let shape = &fcaps[1];
                let fill_val = remove_kwargs(&fcaps[2]);
                return format!(
                    "let {} = Variable::new(Tensor::full(&[{}], {}));",
                    var, shape, fill_val.trim()
                );
            }
        }

        // int(...) casts
        let re_int = Regex::new(r"^int\((.+)\)$").unwrap();
        if let Some(icaps) = re_int.captures(val) {
            return format!("let {} = {} as usize;", var, &icaps[1]);
        }

        // Simple numeric literal
        if val.parse::<f64>().is_ok() {
            return format!("let {} = {};", var, val);
        }

        // Method calls: var = something.method()
        if val.contains(".backward()") {
            return format!("{}.backward();", val.replace(".backward()", ""));
        }
        if val.contains(".step()") {
            return format!("{}.step();", val.replace(".step()", ""));
        }
        if val.contains(".zero_grad()") {
            return format!("{}.zero_grad();", val.replace(".zero_grad()", ""));
        }

        // model(x) -> model.forward(&x) for nn.Module calls
        // Matches: var = modelName(arg) or var = modelName(arg.method())
        let re_model_call = Regex::new(r"^([A-Za-z]\w*)\((.+)\)$").unwrap();
        if let Some(mcaps) = re_model_call.captures(val) {
            let callee = &mcaps[1];
            let args = &mcaps[2];
            // Known patterns: criterion(a, b) -> criterion.forward(&a, &b)
            // model(x) -> model.forward(&x)
            if callee == "criterion" || callee.starts_with("loss") {
                let translated_args = args
                    .split(',')
                    .map(|a| format!("&{}", a.trim()))
                    .collect::<Vec<_>>()
                    .join(", ");
                return format!("let {} = {}.forward({});", var, callee, translated_args);
            }
            // General model call: netG(noise) -> netG.forward(&noise)
            let arg = args.trim();
            return format!("let {} = {}.forward(&{});", var, callee, arg);
        }

        // Default: let var = val;
        let val = python_strings_to_rust(val);
        return format!("let {} = {}; // TODO: verify", var, val);
    }

    // torchvision utils calls
    if trimmed.contains("vutils.") || trimmed.contains("torchvision.") {
        return format!("// TODO: {} (torchvision has no Neo Theano equivalent)", trimmed);
    }

    // Method calls without assignment
    if trimmed.contains(".backward()") {
        return format!("{};", trimmed);
    }
    if trimmed.contains(".step()") {
        return format!("{};", trimmed);
    }
    if trimmed.contains(".zero_grad()") {
        return format!("{};", trimmed);
    }
    if trimmed.contains(".fill_(") {
        return format!("// TODO: {};", trimmed);
    }

    // Fallback — convert any remaining Python strings
    let rust_line = python_strings_to_rust(trimmed);
    format!("// TODO: {}", rust_line)
}

/// Translate a Python `'format' % (args)` or `'format' % var` expression
/// to a Rust `format!("format", args)` expression. If the input doesn't match
/// the pattern, apply `python_strings_to_rust` and return as-is.
fn translate_python_format_string(py: &str) -> String {
    let trimmed = py.trim();

    // Match 'format' % (args) or "format" % (args)
    let re_pct = Regex::new(r#"^['"](.+?)['"]\s*%\s*\((.+)\)$"#).unwrap();
    if let Some(caps) = re_pct.captures(trimmed) {
        let fmt_str = &caps[1];
        let args = &caps[2];
        let rust_fmt = fmt_str
            .replace("%s", "{}")
            .replace("%d", "{}")
            .replace("%03d", "{:03}")
            .replace("%.4f", "{:.4}")
            .replace("%.2f", "{:.2}")
            .replace("%f", "{}");
        return format!("format!(\"{}\", {})", rust_fmt, args);
    }

    // Match 'format' % single_var
    let re_pct_single = Regex::new(r#"^['"](.+?)['"]\s*%\s*(\w+)$"#).unwrap();
    if let Some(caps) = re_pct_single.captures(trimmed) {
        let fmt_str = &caps[1];
        let var = &caps[2];
        let rust_fmt = fmt_str
            .replace("%s", "{}")
            .replace("%d", "{}")
            .replace("%03d", "{:03}")
            .replace("%.4f", "{:.4}")
            .replace("%.2f", "{:.2}")
            .replace("%f", "{}");
        return format!("format!(\"{}\", {})", rust_fmt, var);
    }

    python_strings_to_rust(trimmed)
}

/// Convert Python single-quoted string literals to Rust double-quoted strings.
/// Handles `'text'` -> `"text"` and `''` -> `""`, while preserving char-like
/// single chars used in contexts like `.split(',')`.
fn python_strings_to_rust(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\'' {
            let start = i;
            i += 1;
            let mut content = String::new();
            while i < chars.len() && chars[i] != '\'' {
                if chars[i] == '\\' && i + 1 < chars.len() {
                    content.push(chars[i]);
                    content.push(chars[i + 1]);
                    i += 2;
                } else {
                    content.push(chars[i]);
                    i += 1;
                }
            }
            if i < chars.len() {
                i += 1; // skip closing '
                // Keep as single-quoted only for single chars in split()/find() context
                if content.len() == 1 && start > 0 {
                    let before: String = chars[..start].iter().collect();
                    if before.ends_with("split(") || before.ends_with("find(") {
                        out.push('\'');
                        out.push_str(&content);
                        out.push('\'');
                        continue;
                    }
                }
                out.push('"');
                out.push_str(&content);
                out.push('"');
            } else {
                out.push('\'');
                out.push_str(&content);
            }
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// Translate a Python condition to Rust.
fn translate_condition(py: &str) -> String {
    let mut result = py.to_string();

    // Handle `x in ['a', 'b', 'c']` -> `["a", "b", "c"].contains(&x)`
    let re_in_list = Regex::new(r"(\S+)\s+in\s+\[([^\]]+)\]").unwrap();
    if let Some(caps) = re_in_list.captures(&result.clone()) {
        let var = caps[1].to_string();
        let items = caps[2].to_string();
        let rust_items = python_strings_to_rust(&items);
        result = re_in_list
            .replace(&result, format!("[{}].contains(&{})", rust_items, var))
            .to_string();
    }

    result = result
        .replace(" and ", " && ")
        .replace(" or ", " || ")
        .replace(" is None", ".is_none()")
        .replace(" is not None", ".is_some()")
        .replace(" not ", " !")
        .replace("True", "true")
        .replace("False", "false")
        .replace("None", "None");

    python_strings_to_rust(&result)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Layer Construction — nn.* → Rust type constructors
    // =========================================================================

    #[test]
    fn test_translate_layer_linear() {
        assert_eq!(
            translate_layer_construction("nn.Linear(784, 10)"),
            "Linear::new(784, 10)"
        );
        assert_eq!(
            translate_layer_construction("nn.Linear(in_features, out_features)"),
            "Linear::new(in_features, out_features)"
        );
    }

    #[test]
    fn test_translate_layer_conv2d() {
        // Full args: in_ch, out_ch, kernel, stride, padding, bias
        assert_eq!(
            translate_layer_construction("nn.Conv2d(3, 64, 3, 1, 1, bias=False)"),
            "Conv2d::with_options(3, 64, (3, 3), (1, 1), (1, 1), false)"
        );
        // bias=True
        assert_eq!(
            translate_layer_construction("nn.Conv2d(3, 64, 3, 1, 1, bias=True)"),
            "Conv2d::with_options(3, 64, (3, 3), (1, 1), (1, 1), true)"
        );
        // Without bias kwarg (default true)
        assert_eq!(
            translate_layer_construction("nn.Conv2d(3, 64, 3, 1, 1)"),
            "Conv2d::with_options(3, 64, (3, 3), (1, 1), (1, 1), true)"
        );
        // Only kernel size, no stride/padding
        assert_eq!(
            translate_layer_construction("nn.Conv2d(3, 64, 3)"),
            "Conv2d::with_options(3, 64, (3, 3), (1, 1), (0, 0), true)"
        );
        // Expression args (ngf * 8)
        assert_eq!(
            translate_layer_construction("nn.Conv2d(nz, ngf * 8, 4, 1, 0, bias=False)"),
            "Conv2d::with_options(nz, ngf * 8, (4, 4), (1, 1), (0, 0), false)"
        );
    }

    #[test]
    fn test_translate_layer_conv_transpose2d() {
        // ConvTranspose2d falls back to Conv2d (with TODO in the code)
        assert_eq!(
            translate_layer_construction("nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)"),
            "Conv2d::with_options(nz, ngf * 8, (4, 4), (1, 1), (0, 0), false)"
        );
    }

    #[test]
    fn test_translate_layer_batchnorm() {
        assert_eq!(
            translate_layer_construction("nn.BatchNorm2d(64)"),
            "BatchNorm1d::new(64)"
        );
        assert_eq!(
            translate_layer_construction("nn.BatchNorm1d(128)"),
            "BatchNorm1d::new(128)"
        );
        // Expression args
        assert_eq!(
            translate_layer_construction("nn.BatchNorm2d(ngf * 8)"),
            "BatchNorm1d::new(ngf * 8)"
        );
    }

    #[test]
    fn test_translate_layer_dropout() {
        assert_eq!(
            translate_layer_construction("nn.Dropout(0.5)"),
            "Dropout::new(0.5)"
        );
        assert_eq!(
            translate_layer_construction("nn.Dropout(0.25)"),
            "Dropout::new(0.25)"
        );
    }

    #[test]
    fn test_translate_layer_maxpool2d() {
        assert_eq!(
            translate_layer_construction("nn.MaxPool2d(2)"),
            "MaxPool2d::new(2)"
        );
    }

    #[test]
    fn test_translate_layer_embedding() {
        assert_eq!(
            translate_layer_construction("nn.Embedding(10000, 256)"),
            "Embedding::new(10000, 256)"
        );
        assert_eq!(
            translate_layer_construction("nn.Embedding(vocab_size, embed_dim)"),
            "Embedding::new(vocab_size, embed_dim)"
        );
    }

    #[test]
    fn test_translate_layer_activations() {
        assert_eq!(translate_layer_construction("nn.ReLU(True)"), "ReLU");
        assert_eq!(translate_layer_construction("nn.ReLU()"), "ReLU");
        assert_eq!(translate_layer_construction("nn.ReLU"), "ReLU");
        assert_eq!(translate_layer_construction("nn.Sigmoid()"), "Sigmoid");
        assert_eq!(translate_layer_construction("nn.Tanh()"), "Tanh");
    }

    #[test]
    fn test_translate_layer_leaky_relu() {
        let result = translate_layer_construction("nn.LeakyReLU(0.2, inplace=True)");
        assert!(result.contains("ReLU"));
        assert!(result.contains("LeakyReLU(0.2)"));
    }

    #[test]
    fn test_translate_layer_sequential_empty() {
        assert_eq!(
            translate_layer_construction("nn.Sequential()"),
            "Sequential::new(vec![])"
        );
    }

    #[test]
    fn test_translate_layer_sequential_with_layers() {
        let result = translate_layer_construction(
            "nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))"
        );
        assert!(result.contains("Sequential::new(vec![])"));
        assert!(result.contains(".add(Linear::new(784, 256))"));
        assert!(result.contains(".add(ReLU)"));
        assert!(result.contains(".add(Linear::new(256, 10))"));
    }

    #[test]
    fn test_translate_layer_sequential_with_conv_layers() {
        let result = translate_layer_construction(
            "nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))"
        );
        assert!(result.contains("Sequential::new(vec![])"));
        assert!(result.contains(".add(Conv2d::with_options(3, 64, (3, 3), (1, 1), (1, 1), false))"));
        assert!(result.contains(".add(BatchNorm1d::new(64))"));
        assert!(result.contains(".add(ReLU)"));
    }

    #[test]
    fn test_translate_layer_unknown_fallback() {
        let result = translate_layer_construction("nn.TransformerEncoder(...)");
        assert!(result.contains("TODO"));
    }

    // =========================================================================
    // Expression Translation — tensor creation, ops, method calls
    // =========================================================================

    #[test]
    fn test_translate_expression_self_forward() {
        let fields = vec![("main".to_string(), "Sequential".to_string())];
        assert_eq!(
            translate_expression("self.main(input)", &fields),
            "self.main.forward(&input)"
        );
    }

    #[test]
    fn test_translate_expression_self_layer_call() {
        let fields = vec![
            ("fc1".to_string(), "Linear".to_string()),
            ("fc2".to_string(), "Linear".to_string()),
        ];
        assert_eq!(
            translate_expression("self.fc1(x)", &fields),
            "self.fc1.forward(&x)"
        );
        assert_eq!(
            translate_expression("self.fc2(x)", &fields),
            "self.fc2.forward(&x)"
        );
    }

    #[test]
    fn test_translate_expression_non_field_call() {
        let fields = vec![("fc1".to_string(), "Linear".to_string())];
        // self.unknown_method should not match field-based forward translation
        let result = translate_expression("self.unknown(x)", &fields);
        assert!(result.contains("TODO") || !result.contains(".forward"));
    }

    #[test]
    fn test_translate_expression_view() {
        let fields: Vec<(String, String)> = vec![];
        assert_eq!(
            translate_expression("x.view(-1, 784)", &fields),
            "x.reshape(&[-1, 784]).unwrap()"
        );
        assert_eq!(
            translate_expression("x.view(batch_size, -1)", &fields),
            "x.reshape(&[batch_size, -1]).unwrap()"
        );
    }

    #[test]
    fn test_translate_expression_view_squeeze_chain() {
        let fields: Vec<(String, String)> = vec![];
        assert_eq!(
            translate_expression("output.view(-1, 1).squeeze(1)", &fields),
            "output.reshape(&[-1, 1]).unwrap()"
        );
    }

    #[test]
    fn test_translate_expression_activations() {
        let fields: Vec<(String, String)> = vec![];
        assert_eq!(
            translate_expression("x.relu()", &fields),
            "x.relu().unwrap()"
        );
        assert_eq!(
            translate_expression("x.sigmoid()", &fields),
            "x.sigmoid().unwrap()"
        );
        assert_eq!(
            translate_expression("x.tanh()", &fields),
            "x.tanh().unwrap()"
        );
    }

    #[test]
    fn test_translate_expression_torch_randn() {
        let fields: Vec<(String, String)> = vec![];
        assert_eq!(
            translate_expression("torch.randn(batch_size, nz, 1, 1)", &fields),
            "Variable::new(Tensor::randn(&[batch_size, nz, 1, 1]))"
        );
    }

    #[test]
    fn test_translate_expression_torch_randn_with_device() {
        let fields: Vec<(String, String)> = vec![];
        assert_eq!(
            translate_expression("torch.randn(batch_size, nz, 1, 1, device=device)", &fields),
            "Variable::new(Tensor::randn(&[batch_size, nz, 1, 1]))"
        );
    }

    #[test]
    fn test_translate_expression_torch_zeros() {
        let fields: Vec<(String, String)> = vec![];
        assert_eq!(
            translate_expression("torch.zeros(10, 20)", &fields),
            "Variable::new(Tensor::zeros(&[10, 20]))"
        );
    }

    #[test]
    fn test_translate_expression_torch_ones() {
        let fields: Vec<(String, String)> = vec![];
        assert_eq!(
            translate_expression("torch.ones(5)", &fields),
            "Variable::new(Tensor::ones(&[5]))"
        );
    }

    #[test]
    fn test_translate_expression_torch_full() {
        let fields: Vec<(String, String)> = vec![];
        assert_eq!(
            translate_expression("torch.full((batch_size,), 1.0)", &fields),
            "Variable::new(Tensor::full(&[batch_size,], 1.0))"
        );
    }

    #[test]
    fn test_translate_expression_torch_full_with_device() {
        let fields: Vec<(String, String)> = vec![];
        let result = translate_expression("torch.full((batch_size,), 0.0, device=device)", &fields);
        assert!(result.contains("Tensor::full"));
        assert!(result.contains("batch_size"));
        assert!(result.contains("0.0"));
        // device kwarg should be stripped
        assert!(!result.contains("device=device"));
    }

    // =========================================================================
    // Statement Translation — assignment, control flow, training
    // =========================================================================

    #[test]
    fn test_translate_statement_print() {
        let result = translate_statement("print('hello')");
        assert!(result.contains("println!"));
        assert!(result.contains("\"hello\""));
    }

    #[test]
    fn test_translate_statement_print_format_string() {
        let result = translate_statement(
            "print('[%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, niter, errD.item(), errG.item()))"
        );
        assert!(result.contains("println!"));
        assert!(result.contains("{:.4}"));
        assert!(result.contains("epoch"));
    }

    #[test]
    fn test_translate_statement_model_instantiation() {
        assert_eq!(
            translate_statement("netG = Generator(ngpu)"),
            "let netG = Generator::new(ngpu);"
        );
        assert_eq!(
            translate_statement("netD = Discriminator(ngpu)"),
            "let netD = Discriminator::new(ngpu);"
        );
    }

    #[test]
    fn test_translate_statement_model_with_to_device() {
        assert_eq!(
            translate_statement("model = MyNet(10).to(device)"),
            "let model = MyNet::new(10);"
        );
    }

    #[test]
    fn test_translate_statement_loss_bce() {
        assert_eq!(
            translate_statement("criterion = nn.BCELoss()"),
            "let criterion = BCELoss::new();"
        );
    }

    #[test]
    fn test_translate_statement_loss_mse() {
        assert_eq!(
            translate_statement("criterion = nn.MSELoss()"),
            "let criterion = MSELoss::new();"
        );
    }

    #[test]
    fn test_translate_statement_loss_cross_entropy() {
        assert_eq!(
            translate_statement("criterion = nn.CrossEntropyLoss()"),
            "let criterion = CrossEntropyLoss::new();"
        );
    }

    #[test]
    fn test_translate_statement_optim_adam() {
        let result = translate_statement(
            "optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))"
        );
        assert!(result.contains("let mut optimizerG = Adam::new(netG.parameters(), 0.0002)"));
        assert!(result.contains(".betas(0.5, 0.999)"));
    }

    #[test]
    fn test_translate_statement_optim_sgd() {
        let result = translate_statement(
            "optimizer = optim.SGD(model.parameters(), lr=0.01)"
        );
        assert!(result.contains("let mut optimizer = SGD::new(model.parameters(), 0.01)"));
    }

    #[test]
    fn test_translate_statement_backward() {
        assert_eq!(
            translate_statement("errD.backward()"),
            "errD.backward();"
        );
        assert_eq!(
            translate_statement("loss.backward()"),
            "loss.backward();"
        );
    }

    #[test]
    fn test_translate_statement_optimizer_step() {
        assert_eq!(
            translate_statement("optimizer.step()"),
            "optimizer.step();"
        );
    }

    #[test]
    fn test_translate_statement_zero_grad() {
        assert_eq!(
            translate_statement("optimizer.zero_grad()"),
            "optimizer.zero_grad();"
        );
        assert_eq!(
            translate_statement("netD.zero_grad()"),
            "netD.zero_grad();"
        );
    }

    #[test]
    fn test_translate_statement_for_range() {
        assert_eq!(
            translate_statement("for epoch in range(niter):"),
            "for epoch in 0..niter {"
        );
        assert_eq!(
            translate_statement("for i in range(100):"),
            "for i in 0..100 {"
        );
    }

    #[test]
    fn test_translate_statement_for_enumerate() {
        assert_eq!(
            translate_statement("for i, data in enumerate(dataloader, 0):"),
            "for (i, data) in dataloader.iter().enumerate() {"
        );
    }

    #[test]
    fn test_translate_statement_if_condition() {
        assert_eq!(
            translate_statement("if epoch > 10:"),
            "if epoch > 10 {"
        );
    }

    #[test]
    fn test_translate_statement_elif() {
        assert_eq!(
            translate_statement("elif x > 5:"),
            "} else if x > 5 {"
        );
    }

    #[test]
    fn test_translate_statement_else() {
        assert_eq!(
            translate_statement("else:"),
            "} else {"
        );
    }

    #[test]
    fn test_translate_statement_break() {
        assert_eq!(translate_statement("break"), "break;");
    }

    #[test]
    fn test_translate_statement_assert() {
        assert_eq!(
            translate_statement("assert x > 0"),
            "assert!(x > 0);"
        );
    }

    #[test]
    fn test_translate_statement_torch_randn_assignment() {
        let result = translate_statement(
            "noise = torch.randn(batch_size, nz, 1, 1, device=device)"
        );
        assert!(result.contains("let noise = Variable::new(Tensor::randn(&[batch_size, nz, 1, 1]))"));
        assert!(!result.contains("device=device"));
    }

    #[test]
    fn test_translate_statement_torch_full_assignment() {
        let result = translate_statement(
            "label = torch.full((batch_size,), 1.0, device=device)"
        );
        assert!(result.contains("let label = Variable::new(Tensor::full(&[batch_size,], 1.0))"));
    }

    #[test]
    fn test_translate_statement_int_cast() {
        assert_eq!(
            translate_statement("n = int(x)"),
            "let n = x as usize;"
        );
    }

    #[test]
    fn test_translate_statement_numeric_literal() {
        assert_eq!(
            translate_statement("lr = 0.001"),
            "let lr = 0.001;"
        );
        assert_eq!(
            translate_statement("batch_size = 64"),
            "let batch_size = 64;"
        );
    }

    #[test]
    fn test_translate_statement_model_forward_call() {
        let result = translate_statement("output = netD(real)");
        assert!(result.contains("let output = netD.forward(&real)"));
    }

    #[test]
    fn test_translate_statement_criterion_call() {
        let result = translate_statement("errD_real = criterion(output, label)");
        assert!(result.contains("let errD_real = criterion.forward(&output, &label)"));
    }

    #[test]
    fn test_translate_statement_torch_save() {
        let result = translate_statement(
            "torch.save(netG.state_dict(), 'checkpoint.pt')"
        );
        assert!(result.contains("theano_serialize::save_state_dict(&netG.state_dict())"));
        assert!(result.contains("std::fs::write"));
    }

    #[test]
    fn test_translate_statement_torch_load() {
        let result = translate_statement(
            "netG.load_state_dict(torch.load('checkpoint.pt'))"
        );
        assert!(result.contains("theano_serialize::load_state_dict"));
        assert!(result.contains("std::fs::read"));
    }

    #[test]
    fn test_translate_statement_model_apply() {
        let result = translate_statement("netG.apply(weights_init)");
        assert!(result.contains("TODO"));
    }

    #[test]
    fn test_translate_statement_pass() {
        assert_eq!(translate_statement("pass"), "");
    }

    #[test]
    fn test_translate_statement_from_future() {
        assert_eq!(translate_statement("from __future__ import print_function"), "");
    }

    #[test]
    fn test_translate_statement_try_except() {
        assert!(translate_statement("try:").contains("TODO"));
        assert!(translate_statement("except Exception as e:").contains("TODO"));
        assert!(translate_statement("raise ValueError('error')").contains("TODO"));
    }

    #[test]
    fn test_translate_statement_torchvision_call() {
        let result = translate_statement("vutils.save_image(fake, 'output.png')");
        assert!(result.contains("TODO"));
        assert!(result.contains("torchvision"));
    }

    #[test]
    fn test_translate_statement_fill_() {
        let result = translate_statement("label.fill_(1.0)");
        assert!(result.contains("TODO"));
    }

    // =========================================================================
    // Condition Translation — boolean operators, comparisons
    // =========================================================================

    #[test]
    fn test_translate_condition_in_list() {
        let result = translate_condition("opt.dataset in ['imagenet', 'folder', 'lfw']");
        assert!(result.contains("[\"imagenet\", \"folder\", \"lfw\"].contains(&opt.dataset)"));
    }

    #[test]
    fn test_translate_condition_string_comparison() {
        assert_eq!(
            translate_condition("opt.dataset == 'fake'"),
            "opt.dataset == \"fake\""
        );
    }

    #[test]
    fn test_translate_condition_empty_string() {
        assert_eq!(
            translate_condition("opt.netG != ''"),
            "opt.netG != \"\""
        );
    }

    #[test]
    fn test_translate_condition_and() {
        let result = translate_condition("x > 0 and y > 0");
        assert_eq!(result, "x > 0 && y > 0");
    }

    #[test]
    fn test_translate_condition_or() {
        let result = translate_condition("x > 0 or y > 0");
        assert_eq!(result, "x > 0 || y > 0");
    }

    #[test]
    fn test_translate_condition_is_none() {
        assert_eq!(
            translate_condition("x is None"),
            "x.is_none()"
        );
    }

    #[test]
    fn test_translate_condition_is_not_none() {
        assert_eq!(
            translate_condition("x is not None"),
            "x.is_some()"
        );
    }

    #[test]
    fn test_translate_condition_true_false() {
        assert_eq!(translate_condition("True"), "true");
        assert_eq!(translate_condition("False"), "false");
    }

    #[test]
    fn test_translate_condition_not() {
        let result = translate_condition("x not in list");
        assert!(result.contains("!"));
    }

    #[test]
    fn test_translate_condition_compound() {
        let result = translate_condition("x > 0 and y is not None or z == 'test'");
        assert!(result.contains("&&"));
        assert!(result.contains(".is_some()"));
        assert!(result.contains("||"));
        assert!(result.contains("\"test\""));
    }

    // =========================================================================
    // Python Strings to Rust — quote conversion
    // =========================================================================

    #[test]
    fn test_python_strings_to_rust_basic() {
        assert_eq!(python_strings_to_rust("'fake'"), "\"fake\"");
        assert_eq!(python_strings_to_rust("''"), "\"\"");
        assert_eq!(python_strings_to_rust("'imagenet'"), "\"imagenet\"");
    }

    #[test]
    fn test_python_strings_to_rust_multiword() {
        assert_eq!(
            python_strings_to_rust("'hello world'"),
            "\"hello world\""
        );
    }

    #[test]
    fn test_python_strings_to_rust_split_context() {
        assert_eq!(python_strings_to_rust("split(',')"), "split(',')");
        assert_eq!(python_strings_to_rust("find(':')"), "find(':')");
    }

    #[test]
    fn test_python_strings_to_rust_double_quotes_unchanged() {
        assert_eq!(python_strings_to_rust("\"hello\""), "\"hello\"");
    }

    #[test]
    fn test_python_strings_to_rust_mixed_content() {
        let result = python_strings_to_rust("x == 'test' and y == 'foo'");
        assert_eq!(result, "x == \"test\" and y == \"foo\"");
    }

    #[test]
    fn test_python_strings_to_rust_escaped_quote() {
        let result = python_strings_to_rust("'it\\'s a test'");
        assert!(result.contains("\"it\\'s a test\""));
    }

    #[test]
    fn test_python_strings_to_rust_no_quotes() {
        assert_eq!(python_strings_to_rust("x + y"), "x + y");
    }

    // =========================================================================
    // Format String Translation — Python % format → Rust format!()
    // =========================================================================

    #[test]
    fn test_translate_format_string_pct_tuple() {
        let result = translate_python_format_string("'%s/model_%d.pt' % (outf, epoch)");
        assert_eq!(result, "format!(\"{}/model_{}.pt\", outf, epoch)");
    }

    #[test]
    fn test_translate_format_string_pct_single() {
        let result = translate_python_format_string("'checkpoint_%d.pt' % epoch");
        assert_eq!(result, "format!(\"checkpoint_{}.pt\", epoch)");
    }

    #[test]
    fn test_translate_format_string_float_precision() {
        let result = translate_python_format_string("'loss: %.4f' % loss_val");
        assert_eq!(result, "format!(\"loss: {:.4}\", loss_val)");
    }

    #[test]
    fn test_translate_format_string_no_pattern() {
        // Non-format strings should pass through python_strings_to_rust
        let result = translate_python_format_string("'checkpoint.pt'");
        assert_eq!(result, "\"checkpoint.pt\"");
    }

    #[test]
    fn test_translate_format_string_multiple_specifiers() {
        let result = translate_python_format_string(
            "'%s/epoch_%d_loss_%.4f.pt' % (outf, epoch, loss)"
        );
        assert_eq!(result, "format!(\"{}/epoch_{}_loss_{:.4}.pt\", outf, epoch, loss)");
    }

    // =========================================================================
    // Remove Kwargs — strip keyword arguments from function calls
    // =========================================================================

    #[test]
    fn test_remove_kwargs_device() {
        assert_eq!(
            remove_kwargs("batch_size, nz, 1, 1, device=device"),
            "batch_size, nz, 1, 1"
        );
    }

    #[test]
    fn test_remove_kwargs_multiple() {
        assert_eq!(
            remove_kwargs("10, 20, device=cuda, dtype=float32"),
            "10, 20"
        );
    }

    #[test]
    fn test_remove_kwargs_no_kwargs() {
        assert_eq!(
            remove_kwargs("10, 20, 30"),
            "10, 20, 30"
        );
    }

    #[test]
    fn test_remove_kwargs_only_kwargs() {
        assert_eq!(
            remove_kwargs("device=cuda"),
            ""
        );
    }

    // =========================================================================
    // Forward Body Translation — nn.Module forward methods
    // =========================================================================

    #[test]
    fn test_translate_forward_simple_return() {
        let method = PyMethod {
            name: "forward".to_string(),
            args: vec!["self".to_string(), "x".to_string()],
            body: vec!["return self.fc(x)".to_string()],
            line: 1,
        };
        let fields = vec![("fc".to_string(), "Linear".to_string())];
        let result = translate_forward_body(&method, &fields);
        assert!(result.iter().any(|l| l.contains("self.fc.forward")));
    }

    #[test]
    fn test_translate_forward_two_lines() {
        let method = PyMethod {
            name: "forward".to_string(),
            args: vec!["self".to_string(), "x".to_string()],
            body: vec![
                "x = self.fc1(x)".to_string(),
                "return self.fc2(x)".to_string(),
            ],
            line: 1,
        };
        let fields = vec![
            ("fc1".to_string(), "Linear".to_string()),
            ("fc2".to_string(), "Linear".to_string()),
        ];
        let result = translate_forward_body(&method, &fields);
        assert!(result.iter().any(|l| l.contains("self.fc1.forward")));
        assert!(result.iter().any(|l| l.contains("self.fc2.forward")));
    }

    #[test]
    fn test_translate_forward_with_data_parallel() {
        let method = PyMethod {
            name: "forward".to_string(),
            args: vec!["self".to_string(), "input".to_string()],
            body: vec![
                "if input.is_cuda and self.ngpu > 1:".to_string(),
                "    output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))".to_string(),
                "else:".to_string(),
                "    output = self.main(input)".to_string(),
                "return output".to_string(),
            ],
            line: 1,
        };
        let fields = vec![("main".to_string(), "Sequential".to_string())];
        let result = translate_forward_body(&method, &fields);
        // Should simplify data_parallel to just self.main.forward
        assert!(result.iter().any(|l| l.contains("self.main.forward")));
    }

    #[test]
    fn test_translate_forward_with_view_squeeze() {
        let method = PyMethod {
            name: "forward".to_string(),
            args: vec!["self".to_string(), "input".to_string()],
            body: vec![
                "if input.is_cuda and self.ngpu > 1:".to_string(),
                "    output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))".to_string(),
                "else:".to_string(),
                "    output = self.main(input)".to_string(),
                "return output.view(-1, 1).squeeze(1)".to_string(),
            ],
            line: 1,
        };
        let fields = vec![("main".to_string(), "Sequential".to_string())];
        let result = translate_forward_body(&method, &fields);
        // Should contain reshape
        assert!(result.iter().any(|l| l.contains("reshape")));
    }

    // =========================================================================
    // Forward Statement Translation
    // =========================================================================

    #[test]
    fn test_translate_forward_statement_return() {
        let fields = vec![("fc".to_string(), "Linear".to_string())];
        let result = translate_forward_statement("return self.fc(x)", &fields);
        assert_eq!(result, "self.fc.forward(&x)");
    }

    #[test]
    fn test_translate_forward_statement_assignment() {
        let fields = vec![("conv".to_string(), "Conv2d".to_string())];
        let result = translate_forward_statement("x = self.conv(input)", &fields);
        assert_eq!(result, "let x = self.conv.forward(&input);");
    }

    #[test]
    fn test_translate_forward_statement_fallback() {
        let fields: Vec<(String, String)> = vec![];
        let result = translate_forward_statement("x = something_unknown()", &fields);
        assert!(result.contains("TODO") || result.contains("let x"));
    }

    // =========================================================================
    // Module Struct Generation — full class → struct + impl
    // =========================================================================

    #[test]
    fn test_generate_module_struct_simple() {
        let class = PyClass {
            name: "SimpleNet".to_string(),
            bases: vec!["nn.Module".to_string()],
            methods: vec![
                PyMethod {
                    name: "__init__".to_string(),
                    args: vec!["self".to_string()],
                    body: vec![
                        "super().__init__()".to_string(),
                        "self.fc = nn.Linear(784, 10)".to_string(),
                    ],
                    line: 1,
                },
                PyMethod {
                    name: "forward".to_string(),
                    args: vec!["self".to_string(), "x".to_string()],
                    body: vec!["return self.fc(x)".to_string()],
                    line: 5,
                },
            ],
            body_lines: vec![],
            line: 1,
        };
        let mut changes = Vec::new();
        let result = generate_module_struct(&class, &mut changes, "test.py");

        assert!(result.contains("struct SimpleNet"));
        assert!(result.contains("fc: Linear"));
        assert!(result.contains("impl SimpleNet"));
        assert!(result.contains("fn new("));
        assert!(result.contains("impl Module for SimpleNet"));
        assert!(result.contains("fn forward(&self, input: &Variable) -> Variable"));
        assert!(result.contains("fn parameters(&self) -> Vec<Variable>"));
    }

    #[test]
    fn test_generate_module_struct_multi_layer() {
        let class = PyClass {
            name: "TwoLayerNet".to_string(),
            bases: vec!["nn.Module".to_string()],
            methods: vec![
                PyMethod {
                    name: "__init__".to_string(),
                    args: vec!["self".to_string(), "input_size".to_string(), "hidden_size".to_string()],
                    body: vec![
                        "super().__init__()".to_string(),
                        "self.fc1 = nn.Linear(input_size, hidden_size)".to_string(),
                        "self.fc2 = nn.Linear(hidden_size, 10)".to_string(),
                    ],
                    line: 1,
                },
                PyMethod {
                    name: "forward".to_string(),
                    args: vec!["self".to_string(), "x".to_string()],
                    body: vec![
                        "x = self.fc1(x)".to_string(),
                        "return self.fc2(x)".to_string(),
                    ],
                    line: 6,
                },
            ],
            body_lines: vec![],
            line: 1,
        };
        let mut changes = Vec::new();
        let result = generate_module_struct(&class, &mut changes, "test.py");

        assert!(result.contains("fc1: Linear"));
        assert!(result.contains("fc2: Linear"));
        assert!(result.contains("fn new(input_size: usize, hidden_size: usize)"));
        // Parameters should combine both layers
        assert!(result.contains("self.fc1.parameters()"));
        assert!(result.contains("self.fc2.parameters()"));
    }

    #[test]
    fn test_generate_module_struct_no_forward() {
        let class = PyClass {
            name: "Stub".to_string(),
            bases: vec!["nn.Module".to_string()],
            methods: vec![
                PyMethod {
                    name: "__init__".to_string(),
                    args: vec!["self".to_string()],
                    body: vec!["super().__init__()".to_string()],
                    line: 1,
                },
            ],
            body_lines: vec![],
            line: 1,
        };
        let mut changes = Vec::new();
        let result = generate_module_struct(&class, &mut changes, "test.py");
        // Should have a TODO placeholder for forward
        assert!(result.contains("TODO: implement forward pass"));
    }

    // =========================================================================
    // Cargo.toml Generation — dependency detection
    // =========================================================================

    #[test]
    fn test_generate_cargo_toml_basic() {
        let pyfile = PyFile {
            imports: vec![],
            classes: vec![],
            functions: vec![],
            top_level: vec![],
        };
        let files = vec![("main.py".to_string(), pyfile)];
        let result = generate_cargo_toml_named(&files, "my_project");

        assert!(result.contains("name = \"my_project\""));
        assert!(result.contains("theano"));
        assert!(result.contains("edition = \"2021\""));
    }

    #[test]
    fn test_generate_cargo_toml_with_argparse() {
        let pyfile = PyFile {
            imports: vec![PyImport {
                kind: ImportKind::Import,
                module: "argparse".to_string(),
                names: vec![],
                alias: None,
                line: 1,
            }],
            classes: vec![],
            functions: vec![],
            top_level: vec![],
        };
        let files = vec![("main.py".to_string(), pyfile)];
        let result = generate_cargo_toml_named(&files, "test_proj");

        assert!(result.contains("clap"));
    }

    #[test]
    fn test_generate_cargo_toml_with_random() {
        let pyfile = PyFile {
            imports: vec![PyImport {
                kind: ImportKind::Import,
                module: "random".to_string(),
                names: vec![],
                alias: None,
                line: 1,
            }],
            classes: vec![],
            functions: vec![],
            top_level: vec![],
        };
        let files = vec![("main.py".to_string(), pyfile)];
        let result = generate_cargo_toml_named(&files, "test_proj");

        assert!(result.contains("rand"));
    }

    #[test]
    fn test_generate_cargo_toml_with_save_load() {
        let pyfile = PyFile {
            imports: vec![],
            classes: vec![],
            functions: vec![],
            top_level: vec![PyStatement {
                text: "torch.save(model.state_dict(), 'model.pt')".to_string(),
                line: 10,
            }],
        };
        let files = vec![("main.py".to_string(), pyfile)];
        let result = generate_cargo_toml_named(&files, "test_proj");

        assert!(result.contains("theano-serialize"));
    }

    // =========================================================================
    // Type Inference — Python layer → Rust type
    // =========================================================================

    #[test]
    fn test_infer_rust_type_all_layers() {
        assert_eq!(infer_rust_type("nn.Sequential(...)"), "Sequential");
        assert_eq!(infer_rust_type("nn.Conv2d(3, 64, 3)"), "Conv2d");
        assert_eq!(infer_rust_type("nn.Conv1d(3, 64, 3)"), "Conv1d");
        assert_eq!(infer_rust_type("nn.ConvTranspose2d(...)"), "Conv2d"); // fallback
        assert_eq!(infer_rust_type("nn.Linear(784, 10)"), "Linear");
        assert_eq!(infer_rust_type("nn.BatchNorm2d(64)"), "BatchNorm1d");
        assert_eq!(infer_rust_type("nn.BatchNorm1d(64)"), "BatchNorm1d");
        assert_eq!(infer_rust_type("nn.Dropout(0.5)"), "Dropout");
        assert_eq!(infer_rust_type("nn.MaxPool2d(2)"), "MaxPool2d");
        assert_eq!(infer_rust_type("nn.AvgPool2d(2)"), "AvgPool2d");
        assert_eq!(infer_rust_type("nn.AdaptiveAvgPool2d(1)"), "AdaptiveAvgPool2d");
        assert_eq!(infer_rust_type("nn.Embedding(10000, 256)"), "Embedding");
        assert_eq!(infer_rust_type("nn.ReLU()"), "ReLU");
        assert_eq!(infer_rust_type("nn.Sigmoid()"), "Sigmoid");
        assert_eq!(infer_rust_type("nn.Tanh()"), "Tanh");
    }

    #[test]
    fn test_infer_rust_type_unknown() {
        let result = infer_rust_type("nn.SomeUnknownLayer(...)");
        assert!(result.contains("TODO"));
    }

    // =========================================================================
    // Layer Detection — is_layer_construction
    // =========================================================================

    #[test]
    fn test_is_layer_construction_positive() {
        assert!(is_layer_construction("nn.Linear(784, 10)"));
        assert!(is_layer_construction("nn.Conv2d(3, 64, 3)"));
        assert!(is_layer_construction("nn.Conv1d(3, 64, 3)"));
        assert!(is_layer_construction("nn.ConvTranspose2d(nz, ngf * 8, 4)"));
        assert!(is_layer_construction("nn.BatchNorm2d(64)"));
        assert!(is_layer_construction("nn.Sequential(nn.Linear(1, 2))"));
        assert!(is_layer_construction("nn.Dropout(0.5)"));
        assert!(is_layer_construction("nn.Embedding(10000, 256)"));
        assert!(is_layer_construction("nn.MaxPool2d(2)"));
        assert!(is_layer_construction("nn.AvgPool2d(2)"));
        assert!(is_layer_construction("nn.AdaptiveAvgPool2d(1)"));
        assert!(is_layer_construction("torch.nn.Linear(10, 20)"));
    }

    #[test]
    fn test_is_layer_construction_negative() {
        assert!(!is_layer_construction("ngpu"));
        assert!(!is_layer_construction("x + y"));
        assert!(!is_layer_construction("0.5"));
        assert!(!is_layer_construction("some_function()"));
    }

    // =========================================================================
    // Extract Layer Fields
    // =========================================================================

    #[test]
    fn test_extract_layer_fields() {
        let init = PyMethod {
            name: "__init__".to_string(),
            args: vec!["self".to_string()],
            body: vec![
                "super().__init__()".to_string(),
                "self.ngpu = ngpu".to_string(), // should be skipped (not a layer)
                "self.fc = nn.Linear(784, 10)".to_string(),
                "self.conv = nn.Conv2d(3, 64, 3)".to_string(),
            ],
            line: 1,
        };
        let fields = extract_layer_fields(&init);
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].0, "fc");
        assert_eq!(fields[0].1, "Linear");
        assert_eq!(fields[1].0, "conv");
        assert_eq!(fields[1].1, "Conv2d");
    }

    #[test]
    fn test_extract_layer_fields_skips_scalars() {
        let init = PyMethod {
            name: "__init__".to_string(),
            args: vec!["self".to_string()],
            body: vec![
                "super().__init__()".to_string(),
                "self.num_classes = num_classes".to_string(),
                "self.hidden_size = 256".to_string(),
            ],
            line: 1,
        };
        let fields = extract_layer_fields(&init);
        assert_eq!(fields.len(), 0);
    }

    // =========================================================================
    // Join Multiline Expressions
    // =========================================================================

    #[test]
    fn test_join_multiline_single_line() {
        let lines = vec!["x = 10".to_string()];
        let result = join_multiline_exprs(&lines);
        assert_eq!(result, vec!["x = 10"]);
    }

    #[test]
    fn test_join_multiline_continuation() {
        let lines = vec![
            "self.main = nn.Sequential(".to_string(),
            "    nn.Linear(784, 256),".to_string(),
            "    nn.ReLU(),".to_string(),
            ")".to_string(),
        ];
        let result = join_multiline_exprs(&lines);
        assert_eq!(result.len(), 1);
        assert!(result[0].contains("nn.Sequential("));
        assert!(result[0].contains("nn.Linear(784, 256)"));
        assert!(result[0].contains("nn.ReLU()"));
    }

    #[test]
    fn test_join_multiline_nested_parens() {
        let lines = vec![
            "self.main = nn.Sequential(".to_string(),
            "    nn.Conv2d(3, 64, 3, 1, 1),".to_string(),
            "    nn.BatchNorm2d(64),".to_string(),
            "    nn.ReLU(True),".to_string(),
            ")".to_string(),
        ];
        let result = join_multiline_exprs(&lines);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_join_multiline_with_comments() {
        let lines = vec![
            "self.main = nn.Sequential(".to_string(),
            "    nn.Linear(784, 256), # first layer".to_string(),
            "    nn.ReLU(), # activation".to_string(),
            ")".to_string(),
        ];
        let result = join_multiline_exprs(&lines);
        assert_eq!(result.len(), 1);
        // Comments should be stripped
        assert!(!result[0].contains('#'));
    }

    // =========================================================================
    // Strip Line Comments
    // =========================================================================

    #[test]
    fn test_strip_line_comment_basic() {
        assert_eq!(strip_line_comment("x = 10 # comment"), "x = 10");
    }

    #[test]
    fn test_strip_line_comment_in_string() {
        assert_eq!(
            strip_line_comment("x = '#not a comment'"),
            "x = '#not a comment'"
        );
    }

    #[test]
    fn test_strip_line_comment_no_comment() {
        assert_eq!(strip_line_comment("x = 10"), "x = 10");
    }

    #[test]
    fn test_strip_line_comment_full_line() {
        assert_eq!(strip_line_comment("# just a comment"), "");
    }

    // =========================================================================
    // Strip Inline Comments
    // =========================================================================

    #[test]
    fn test_strip_inline_comments_multiline() {
        let input = "nn.Linear(10, 20), # first\nnn.ReLU() # second";
        let result = strip_inline_comments(input);
        assert!(!result.contains("first"));
        assert!(!result.contains("second"));
        assert!(result.contains("nn.Linear(10, 20)"));
        assert!(result.contains("nn.ReLU()"));
    }

    // =========================================================================
    // Argparse Extraction
    // =========================================================================

    #[test]
    fn test_extract_argparse_args_basic() {
        let pyfile = PyFile {
            imports: vec![],
            classes: vec![],
            functions: vec![],
            top_level: vec![
                PyStatement {
                    text: "parser = argparse.ArgumentParser()".to_string(),
                    line: 1,
                },
                PyStatement {
                    text: "parser.add_argument('--lr', type=float, default=0.001, help='learning rate')".to_string(),
                    line: 2,
                },
                PyStatement {
                    text: "parser.add_argument('--epochs', type=int, default=10, help='number of epochs')".to_string(),
                    line: 3,
                },
            ],
        };
        let files = vec![("main.py".to_string(), pyfile)];
        let args = extract_argparse_args(&files);
        assert_eq!(args.len(), 2);
        assert_eq!(args[0].rust_name, "lr");
        assert_eq!(args[0].arg_type, "float");
        assert_eq!(args[0].default, Some("0.001".to_string()));
        assert_eq!(args[0].help, Some("learning rate".to_string()));
        assert_eq!(args[1].rust_name, "epochs");
        assert_eq!(args[1].arg_type, "int");
    }

    #[test]
    fn test_extract_argparse_flag() {
        let pyfile = PyFile {
            imports: vec![],
            classes: vec![],
            functions: vec![],
            top_level: vec![PyStatement {
                text: "parser.add_argument('--dry-run', action='store_true', help='run without saving')".to_string(),
                line: 1,
            }],
        };
        let files = vec![("main.py".to_string(), pyfile)];
        let args = extract_argparse_args(&files);
        assert_eq!(args.len(), 1);
        assert_eq!(args[0].rust_name, "dry_run"); // hyphen → underscore
        assert!(args[0].is_flag);
        assert_eq!(args[0].default, Some("false".to_string()));
    }

    #[test]
    fn test_extract_argparse_string_default() {
        let pyfile = PyFile {
            imports: vec![],
            classes: vec![],
            functions: vec![],
            top_level: vec![PyStatement {
                text: "parser.add_argument('--outf', default='output', help='output folder')".to_string(),
                line: 1,
            }],
        };
        let files = vec![("main.py".to_string(), pyfile)];
        let args = extract_argparse_args(&files);
        assert_eq!(args.len(), 1);
        assert_eq!(args[0].default, Some("\"output\"".to_string()));
    }

    // =========================================================================
    // Rust Default Value Conversion
    // =========================================================================

    #[test]
    fn test_rust_default_value_bool() {
        assert_eq!(rust_default_value("false", "bool"), "false");
        assert_eq!(rust_default_value("true", "bool"), "true");
    }

    #[test]
    fn test_rust_default_value_int() {
        assert_eq!(rust_default_value("42", "int"), "42");
    }

    #[test]
    fn test_rust_default_value_float() {
        assert_eq!(rust_default_value("0.001", "float"), "0.001");
        // Integer value for float type gets .0 appended
        assert_eq!(rust_default_value("1", "float"), "1.0");
    }

    #[test]
    fn test_rust_default_value_string() {
        assert_eq!(
            rust_default_value("\"output\"", "str"),
            "\"output\".into()"
        );
        assert_eq!(
            rust_default_value("String::new()", "str"),
            "String::new().into()"
        );
    }

    // =========================================================================
    // Translate Args — adding & references
    // =========================================================================

    #[test]
    fn test_translate_args_variables() {
        assert_eq!(translate_args("x"), "&x");
        assert_eq!(translate_args("x, y"), "&x, &y");
    }

    #[test]
    fn test_translate_args_literals() {
        // Numbers should not get & prefix
        assert_eq!(translate_args("3.14"), "3.14");
        assert_eq!(translate_args("42"), "42");
    }

    #[test]
    fn test_translate_args_already_referenced() {
        assert_eq!(translate_args("&x"), "&x");
    }

    #[test]
    fn test_translate_args_string_literal() {
        assert_eq!(translate_args("\"hello\""), "\"hello\"");
    }

    // =========================================================================
    // Generate Function — standalone Python functions
    // =========================================================================

    #[test]
    fn test_generate_function_no_args() {
        let func = PyFunction {
            name: "setup".to_string(),
            args: vec![],
            body: vec!["print('hello')".to_string()],
            line: 1,
        };
        let result = generate_function(&func);
        assert!(result.contains("fn setup()"));
    }

    #[test]
    fn test_generate_function_with_args() {
        let func = PyFunction {
            name: "weights_init".to_string(),
            args: vec!["m".to_string()],
            body: vec![
                "classname = m.__class__.__name__".to_string(),
                "if classname.find('Conv') != -1:".to_string(),
            ],
            line: 1,
        };
        let result = generate_function(&func);
        assert!(result.contains("fn weights_init(m: /* TODO */)"));
        assert!(result.contains("// TODO:"));
    }

    // =========================================================================
    // Split Sequential Args
    // =========================================================================

    #[test]
    fn test_split_sequential_args_simple() {
        let args = split_sequential_args("nn.Linear(10, 20), nn.ReLU()");
        assert_eq!(args.len(), 2);
        assert_eq!(args[0], "nn.Linear(10, 20)");
        assert_eq!(args[1], "nn.ReLU()");
    }

    #[test]
    fn test_split_sequential_args_nested() {
        let args = split_sequential_args(
            "nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True)"
        );
        assert_eq!(args.len(), 3);
        assert!(args[0].contains("Conv2d"));
        assert!(args[1].contains("BatchNorm2d"));
        assert!(args[2].contains("ReLU"));
    }

    #[test]
    fn test_split_sequential_args_with_comments() {
        let args = split_sequential_args(
            "nn.Linear(10, 20), # layer 1\nnn.ReLU() # activation"
        );
        assert_eq!(args.len(), 2);
    }

    // =========================================================================
    // Main.rs Generation — full pipeline integration
    // =========================================================================

    #[test]
    fn test_generate_main_rs_use_statements() {
        let import_analysis = ImportAnalysis {
            uses_torch: true,
            uses_nn: true,
            uses_optim: true,
            uses_data: true,
            uses_functional: false,
            torchvision_imports: vec![],
            unsupported_imports: vec![],
        };
        let pyfile = PyFile {
            imports: vec![],
            classes: vec![],
            functions: vec![],
            top_level: vec![],
        };
        let files = vec![("main.py".to_string(), pyfile)];
        let mut changes = Vec::new();
        let result = generate_main_rs(&files, &import_analysis, &mut changes);

        assert!(result.contains("use theano::prelude::*;"));
        assert!(result.contains("use theano::nn::*;"));
        assert!(result.contains("use theano::optim::{Adam, SGD, Optimizer};"));
        assert!(result.contains("use theano_types::Device;"));
    }

    #[test]
    fn test_generate_main_rs_with_save_load_imports() {
        let import_analysis = ImportAnalysis {
            uses_torch: false,
            uses_nn: false,
            uses_optim: false,
            uses_data: false,
            uses_functional: false,
            torchvision_imports: vec![],
            unsupported_imports: vec![],
        };
        let pyfile = PyFile {
            imports: vec![],
            classes: vec![],
            functions: vec![],
            top_level: vec![PyStatement {
                text: "torch.save(model.state_dict(), 'model.pt')".to_string(),
                line: 1,
            }],
        };
        let files = vec![("main.py".to_string(), pyfile)];
        let mut changes = Vec::new();
        let result = generate_main_rs(&files, &import_analysis, &mut changes);

        assert!(result.contains("use theano_serialize::{save_state_dict, load_state_dict};"));
    }

    #[test]
    fn test_generate_main_rs_torchvision_warnings() {
        let import_analysis = ImportAnalysis {
            uses_torch: false,
            uses_nn: false,
            uses_optim: false,
            uses_data: false,
            uses_functional: false,
            torchvision_imports: vec![PyImport {
                kind: ImportKind::Import,
                module: "torchvision".to_string(),
                names: vec![],
                alias: None,
                line: 1,
            }],
            unsupported_imports: vec![],
        };
        let pyfile = PyFile {
            imports: vec![],
            classes: vec![],
            functions: vec![],
            top_level: vec![],
        };
        let files = vec![("main.py".to_string(), pyfile)];
        let mut changes = Vec::new();
        let result = generate_main_rs(&files, &import_analysis, &mut changes);

        assert!(result.contains("Manual attention required"));
        assert!(result.contains("torchvision"));
    }

    #[test]
    fn test_generate_main_rs_with_class_and_statements() {
        let import_analysis = ImportAnalysis {
            uses_torch: true,
            uses_nn: true,
            uses_optim: true,
            uses_data: false,
            uses_functional: false,
            torchvision_imports: vec![],
            unsupported_imports: vec![],
        };
        let pyfile = PyFile {
            imports: vec![],
            classes: vec![PyClass {
                name: "Net".to_string(),
                bases: vec!["nn.Module".to_string()],
                methods: vec![
                    PyMethod {
                        name: "__init__".to_string(),
                        args: vec!["self".to_string()],
                        body: vec![
                            "super().__init__()".to_string(),
                            "self.fc = nn.Linear(784, 10)".to_string(),
                        ],
                        line: 2,
                    },
                    PyMethod {
                        name: "forward".to_string(),
                        args: vec!["self".to_string(), "x".to_string()],
                        body: vec!["return self.fc(x)".to_string()],
                        line: 5,
                    },
                ],
                body_lines: vec![],
                line: 1,
            }],
            functions: vec![],
            top_level: vec![
                PyStatement { text: "model = Net()".to_string(), line: 8 },
                PyStatement { text: "criterion = nn.MSELoss()".to_string(), line: 9 },
            ],
        };
        let files = vec![("main.py".to_string(), pyfile)];
        let mut changes = Vec::new();
        let result = generate_main_rs(&files, &import_analysis, &mut changes);

        // Class struct
        assert!(result.contains("struct Net"));
        assert!(result.contains("fc: Linear"));
        // Main function
        assert!(result.contains("fn main()"));
        assert!(result.contains("let model = Net::new()"));
        assert!(result.contains("let criterion = MSELoss::new()"));
    }

    // =========================================================================
    // Full Project Generation — end-to-end
    // =========================================================================

    #[test]
    fn test_generate_project_basic() {
        let pyfile = PyFile {
            imports: vec![PyImport {
                kind: ImportKind::Import,
                module: "torch".to_string(),
                names: vec![],
                alias: None,
                line: 1,
            }],
            classes: vec![],
            functions: vec![],
            top_level: vec![PyStatement {
                text: "x = 42".to_string(),
                line: 2,
            }],
        };
        let files = vec![("main.py".to_string(), pyfile)];
        let result = generate_project_named(&files, "test_project");

        assert!(result.cargo_toml.contains("name = \"test_project\""));
        assert!(result.main_rs.contains("fn main()"));
        assert!(!result.changes.is_empty());
    }

    #[test]
    fn test_generate_project_with_torchvision_warning() {
        let pyfile = PyFile {
            imports: vec![PyImport {
                kind: ImportKind::Import,
                module: "torchvision".to_string(),
                names: vec![],
                alias: None,
                line: 1,
            }],
            classes: vec![],
            functions: vec![],
            top_level: vec![],
        };
        let files = vec![("main.py".to_string(), pyfile)];
        let result = generate_project_named(&files, "test_project");

        // Should have a warning change
        assert!(result.changes.iter().any(|c| {
            c.kind == ChangeKind::Warning && c.description.contains("torchvision")
        }));
    }

    #[test]
    fn test_generate_project_with_distributed_warning() {
        let pyfile = PyFile {
            imports: vec![PyImport {
                kind: ImportKind::Import,
                module: "torch.distributed".to_string(),
                names: vec![],
                alias: None,
                line: 1,
            }],
            classes: vec![],
            functions: vec![],
            top_level: vec![],
        };
        let files = vec![("main.py".to_string(), pyfile)];
        let result = generate_project_named(&files, "test_project");

        assert!(result.changes.iter().any(|c| {
            c.kind == ChangeKind::Warning && c.description.contains("torch.distributed")
        }));
    }

    // =========================================================================
    // Integration: MLP classifier end-to-end
    // =========================================================================

    #[test]
    fn test_integration_mlp_classifier() {
        let pyfile = PyFile {
            imports: vec![
                PyImport {
                    kind: ImportKind::Import,
                    module: "torch".to_string(),
                    names: vec![],
                    alias: None,
                    line: 1,
                },
                PyImport {
                    kind: ImportKind::FromImport,
                    module: "torch.nn".to_string(),
                    names: vec!["Module".to_string()],
                    alias: None,
                    line: 2,
                },
                PyImport {
                    kind: ImportKind::FromImport,
                    module: "torch.optim".to_string(),
                    names: vec!["Adam".to_string()],
                    alias: None,
                    line: 3,
                },
            ],
            classes: vec![PyClass {
                name: "MLP".to_string(),
                bases: vec!["nn.Module".to_string()],
                methods: vec![
                    PyMethod {
                        name: "__init__".to_string(),
                        args: vec!["self".to_string(), "input_dim".to_string(), "hidden_dim".to_string(), "output_dim".to_string()],
                        body: vec![
                            "super().__init__()".to_string(),
                            "self.fc1 = nn.Linear(input_dim, hidden_dim)".to_string(),
                            "self.fc2 = nn.Linear(hidden_dim, output_dim)".to_string(),
                            "self.dropout = nn.Dropout(0.5)".to_string(),
                        ],
                        line: 5,
                    },
                    PyMethod {
                        name: "forward".to_string(),
                        args: vec!["self".to_string(), "x".to_string()],
                        body: vec![
                            "x = self.fc1(x)".to_string(),
                            "x = self.dropout(x)".to_string(),
                            "return self.fc2(x)".to_string(),
                        ],
                        line: 12,
                    },
                ],
                body_lines: vec![],
                line: 4,
            }],
            functions: vec![],
            top_level: vec![
                PyStatement { text: "model = MLP(784, 256, 10)".to_string(), line: 18 },
                PyStatement { text: "criterion = nn.CrossEntropyLoss()".to_string(), line: 19 },
                PyStatement { text: "optimizer = optim.Adam(model.parameters(), lr=0.001)".to_string(), line: 20 },
                PyStatement { text: "for epoch in range(100):".to_string(), line: 21 },
                PyStatement { text: "    optimizer.zero_grad()".to_string(), line: 22 },
            ],
        };
        let files = vec![("train.py".to_string(), pyfile)];
        let result = generate_project_named(&files, "mlp_classifier");

        // Cargo.toml
        assert!(result.cargo_toml.contains("name = \"mlp_classifier\""));
        assert!(result.cargo_toml.contains("theano"));

        // main.rs structure
        let main = &result.main_rs;
        assert!(main.contains("use theano::prelude::*;"));
        assert!(main.contains("use theano::nn::*;"));
        assert!(main.contains("use theano::optim::{Adam, SGD, Optimizer};"));

        // MLP struct
        assert!(main.contains("struct MLP"));
        assert!(main.contains("fc1: Linear"));
        assert!(main.contains("fc2: Linear"));
        assert!(main.contains("dropout: Dropout"));

        // Constructor
        assert!(main.contains("fn new(input_dim: usize, hidden_dim: usize, output_dim: usize)"));
        assert!(main.contains("Linear::new(input_dim, hidden_dim)"));
        assert!(main.contains("Linear::new(hidden_dim, output_dim)"));
        assert!(main.contains("Dropout::new(0.5)"));

        // Forward
        assert!(main.contains("impl Module for MLP"));
        assert!(main.contains("fn forward(&self, input: &Variable) -> Variable"));

        // Main function
        assert!(main.contains("fn main()"));
        assert!(main.contains("let model = MLP::new(784, 256, 10)"));
        assert!(main.contains("let criterion = CrossEntropyLoss::new()"));
        assert!(main.contains("Adam::new(model.parameters(), 0.001)"));
        assert!(main.contains("for epoch in 0..100 {"));
        assert!(main.contains("zero_grad()"));
    }

    // =========================================================================
    // Integration: CNN with BatchNorm end-to-end
    // =========================================================================

    #[test]
    fn test_integration_cnn_with_batchnorm() {
        let pyfile = PyFile {
            imports: vec![
                PyImport {
                    kind: ImportKind::Import,
                    module: "torch".to_string(),
                    names: vec![],
                    alias: None,
                    line: 1,
                },
                PyImport {
                    kind: ImportKind::Import,
                    module: "torch.nn".to_string(),
                    names: vec![],
                    alias: None,
                    line: 2,
                },
            ],
            classes: vec![PyClass {
                name: "ConvNet".to_string(),
                bases: vec!["nn.Module".to_string()],
                methods: vec![
                    PyMethod {
                        name: "__init__".to_string(),
                        args: vec!["self".to_string()],
                        body: vec![
                            "super().__init__()".to_string(),
                            "self.features = nn.Sequential(".to_string(),
                            "    nn.Conv2d(3, 64, 3, 1, 1, bias=False),".to_string(),
                            "    nn.BatchNorm2d(64),".to_string(),
                            "    nn.ReLU(True),".to_string(),
                            "    nn.MaxPool2d(2),".to_string(),
                            ")".to_string(),
                            "self.classifier = nn.Linear(64, 10)".to_string(),
                        ],
                        line: 3,
                    },
                    PyMethod {
                        name: "forward".to_string(),
                        args: vec!["self".to_string(), "x".to_string()],
                        body: vec![
                            "x = self.features(x)".to_string(),
                            "x = x.view(x.size(0), -1)".to_string(),
                            "return self.classifier(x)".to_string(),
                        ],
                        line: 12,
                    },
                ],
                body_lines: vec![],
                line: 3,
            }],
            functions: vec![],
            top_level: vec![],
        };
        let files = vec![("cnn.py".to_string(), pyfile)];
        let result = generate_project_named(&files, "cnn_example");

        let main = &result.main_rs;

        // Sequential with layers
        assert!(main.contains("features: Sequential"));
        assert!(main.contains("classifier: Linear"));
        assert!(main.contains("Sequential::new(vec![])"));
        assert!(main.contains(".add(Conv2d::with_options(3, 64, (3, 3), (1, 1), (1, 1), false))"));
        assert!(main.contains(".add(BatchNorm1d::new(64))"));
        assert!(main.contains(".add(ReLU)"));
        assert!(main.contains(".add(MaxPool2d::new(2))"));
    }

    // =========================================================================
    // Integration: GAN training loop end-to-end
    // =========================================================================

    #[test]
    fn test_integration_gan_training_loop() {
        let pyfile = PyFile {
            imports: vec![
                PyImport { kind: ImportKind::Import, module: "torch".to_string(), names: vec![], alias: None, line: 1 },
                PyImport { kind: ImportKind::Import, module: "torch.nn".to_string(), names: vec![], alias: None, line: 2 },
                PyImport { kind: ImportKind::Import, module: "torch.optim".to_string(), names: vec![], alias: None, line: 3 },
            ],
            classes: vec![],
            functions: vec![],
            top_level: vec![
                PyStatement { text: "criterion = nn.BCELoss()".to_string(), line: 5 },
                PyStatement { text: "optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))".to_string(), line: 6 },
                PyStatement { text: "optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))".to_string(), line: 7 },
                PyStatement { text: "for epoch in range(25):".to_string(), line: 8 },
                PyStatement { text: "    for i, data in enumerate(dataloader, 0):".to_string(), line: 9 },
                PyStatement { text: "        netD.zero_grad()".to_string(), line: 10 },
                PyStatement { text: "        label = torch.full((64,), 1.0, device=device)".to_string(), line: 11 },
                PyStatement { text: "        output = netD(real)".to_string(), line: 12 },
                PyStatement { text: "        errD_real = criterion(output, label)".to_string(), line: 13 },
                PyStatement { text: "        errD_real.backward()".to_string(), line: 14 },
                PyStatement { text: "        noise = torch.randn(64, 100, 1, 1, device=device)".to_string(), line: 15 },
                PyStatement { text: "        fake = netG(noise)".to_string(), line: 16 },
                PyStatement { text: "        optimizerD.step()".to_string(), line: 17 },
            ],
        };
        let files = vec![("dcgan.py".to_string(), pyfile)];
        let result = generate_project_named(&files, "dcgan");

        let main = &result.main_rs;

        // Loss function
        assert!(main.contains("let criterion = BCELoss::new()"));
        // Optimizers with betas
        assert!(main.contains("Adam::new(netD.parameters(), 0.0002)"));
        assert!(main.contains(".betas(0.5, 0.999)"));
        // Training loop
        assert!(main.contains("for epoch in 0..25 {"));
        assert!(main.contains("for (i, data) in dataloader.iter().enumerate() {"));
        // Training ops
        assert!(main.contains("netD.zero_grad()"));
        assert!(main.contains("Tensor::full(&[64,], 1.0)"));
        assert!(main.contains("netD.forward(&real)"));
        assert!(main.contains("criterion.forward(&output, &label)"));
        assert!(main.contains("errD_real.backward()"));
        assert!(main.contains("Tensor::randn(&[64, 100, 1, 1])"));
        assert!(main.contains("netG.forward(&noise)"));
        assert!(main.contains("optimizerD.step()"));
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_translate_statement_comment_preservation() {
        // Comments in main() should be preserved
        let import_analysis = ImportAnalysis {
            uses_torch: false, uses_nn: false, uses_optim: false, uses_data: false,
            uses_functional: false, torchvision_imports: vec![], unsupported_imports: vec![],
        };
        let pyfile = PyFile {
            imports: vec![],
            classes: vec![],
            functions: vec![],
            top_level: vec![
                PyStatement { text: "# Training loop".to_string(), line: 1 },
                PyStatement { text: "x = 42".to_string(), line: 2 },
            ],
        };
        let files = vec![("main.py".to_string(), pyfile)];
        let mut changes = Vec::new();
        let result = generate_main_rs(&files, &import_analysis, &mut changes);
        assert!(result.contains("// Training loop"));
    }

    #[test]
    fn test_translate_statement_skip_imports() {
        let import_analysis = ImportAnalysis {
            uses_torch: false, uses_nn: false, uses_optim: false, uses_data: false,
            uses_functional: false, torchvision_imports: vec![], unsupported_imports: vec![],
        };
        let pyfile = PyFile {
            imports: vec![],
            classes: vec![],
            functions: vec![],
            top_level: vec![
                PyStatement { text: "import os".to_string(), line: 1 },
                PyStatement { text: "from pathlib import Path".to_string(), line: 2 },
                PyStatement { text: "x = 10".to_string(), line: 3 },
            ],
        };
        let files = vec![("main.py".to_string(), pyfile)];
        let mut changes = Vec::new();
        let result = generate_main_rs(&files, &import_analysis, &mut changes);
        // Import lines should be skipped in main()
        assert!(!result.contains("import os"));
        assert!(!result.contains("from pathlib"));
        // But actual code should be present
        assert!(result.contains("let x = 10"));
    }

    #[test]
    fn test_translate_statement_backward_with_assignment() {
        // errD = something; errD.backward() pattern
        let result = translate_statement("errD.backward()");
        assert_eq!(result, "errD.backward();");
    }

    #[test]
    fn test_empty_pyfile_produces_valid_project() {
        let pyfile = PyFile {
            imports: vec![],
            classes: vec![],
            functions: vec![],
            top_level: vec![],
        };
        let files = vec![("empty.py".to_string(), pyfile)];
        let result = generate_project_named(&files, "empty_project");

        assert!(result.cargo_toml.contains("name = \"empty_project\""));
        assert!(result.main_rs.contains("fn main()"));
        assert!(result.main_rs.contains('}'));
    }
}
