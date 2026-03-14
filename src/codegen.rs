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
    } else if value.contains("AvgPool2d") {
        "AvgPool2d".to_string()
    } else if value.contains("AdaptiveAvgPool2d") {
        "AdaptiveAvgPool2d".to_string()
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

    // x.view(-1, N) -> x.reshape(&[-1, N]).unwrap()
    let re_view = Regex::new(r"^(\w+)\.view\((.+)\)$").unwrap();
    if let Some(caps) = re_view.captures(trimmed) {
        let var = &caps[1];
        let dims = &caps[2];
        return format!("{}.reshape(&[{}]).unwrap()", var, dims);
    }

    // output.view(-1, 1).squeeze(1) pattern
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

    #[test]
    fn test_translate_layer_construction() {
        assert_eq!(
            translate_layer_construction("nn.Linear(784, 10)"),
            "Linear::new(784, 10)"
        );
        assert_eq!(
            translate_layer_construction("nn.Conv2d(3, 64, 3, 1, 1, bias=False)"),
            "Conv2d::with_options(3, 64, (3, 3), (1, 1), (1, 1), false)"
        );
        assert_eq!(
            translate_layer_construction("nn.BatchNorm2d(64)"),
            "BatchNorm1d::new(64)"
        );
    }

    #[test]
    fn test_translate_expression() {
        let fields = vec![("main".to_string(), "Sequential".to_string())];

        assert_eq!(
            translate_expression("self.main(input)", &fields),
            "self.main.forward(&input)"
        );
    }

    #[test]
    fn test_remove_kwargs() {
        assert_eq!(
            remove_kwargs("batch_size, nz, 1, 1, device=device"),
            "batch_size, nz, 1, 1"
        );
    }

    #[test]
    fn test_python_strings_to_rust() {
        assert_eq!(python_strings_to_rust("'fake'"), "\"fake\"");
        assert_eq!(python_strings_to_rust("''"), "\"\"");
        assert_eq!(python_strings_to_rust("'imagenet'"), "\"imagenet\"");
        // Preserve single-char in split() context
        assert_eq!(python_strings_to_rust("split(',')"), "split(',')");
        // Multi-word strings
        assert_eq!(
            python_strings_to_rust("'hello world'"),
            "\"hello world\""
        );
    }

    #[test]
    fn test_translate_condition_in_list() {
        let result = translate_condition("opt.dataset in ['imagenet', 'folder', 'lfw']");
        assert!(result.contains("[\"imagenet\", \"folder\", \"lfw\"].contains(&opt.dataset)"));
    }

    #[test]
    fn test_translate_condition_string_comparison() {
        let result = translate_condition("opt.dataset == 'fake'");
        assert_eq!(result, "opt.dataset == \"fake\"");
    }

    #[test]
    fn test_translate_condition_empty_string() {
        let result = translate_condition("opt.netG != ''");
        assert_eq!(result, "opt.netG != \"\"");
    }
}
