//! Python source analyzer — extracts high-level structure from PyTorch files.
//!
//! Uses regex-based parsing to identify imports, nn.Module classes, functions,
//! and top-level statements without requiring a full Python AST parser.

use regex::Regex;

/// A single Python source file analyzed into structural components.
#[derive(Debug)]
pub struct PyFile {
    pub imports: Vec<PyImport>,
    pub classes: Vec<PyClass>,
    pub functions: Vec<PyFunction>,
    pub top_level: Vec<PyStatement>,
}

#[derive(Debug, Clone)]
pub struct PyImport {
    pub kind: ImportKind,
    pub module: String,
    pub names: Vec<String>,
    pub alias: Option<String>,
    pub line: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ImportKind {
    /// `import torch`
    Import,
    /// `from torch import Tensor`
    FromImport,
}

#[derive(Debug)]
pub struct PyClass {
    pub name: String,
    pub bases: Vec<String>,
    pub methods: Vec<PyMethod>,
    pub body_lines: Vec<String>,
    pub line: usize,
}

#[derive(Debug)]
pub struct PyMethod {
    pub name: String,
    pub args: Vec<String>,
    pub body: Vec<String>,
    pub line: usize,
}

#[derive(Debug)]
pub struct PyFunction {
    pub name: String,
    pub args: Vec<String>,
    pub body: Vec<String>,
    pub line: usize,
}

/// Top-level statement that isn't a class, function, or import.
#[derive(Debug)]
pub struct PyStatement {
    pub text: String,
    pub line: usize,
}

/// Analyze a Python source file into structural components.
pub fn analyze(source: &str) -> PyFile {
    let lines: Vec<&str> = source.lines().collect();
    let mut imports = Vec::new();
    let mut classes = Vec::new();
    let mut functions = Vec::new();
    let mut top_level = Vec::new();

    let re_import = Regex::new(r"^import\s+(\S+)(?:\s+as\s+(\S+))?").unwrap();
    let re_from_import = Regex::new(r"^from\s+(\S+)\s+import\s+(.+)").unwrap();
    let re_class = Regex::new(r"^class\s+(\w+)\s*\(([^)]*)\)\s*:").unwrap();
    let re_def = Regex::new(r"^def\s+(\w+)\s*\(([^)]*)\)\s*(?:->.*)?:").unwrap();
    let re_method = Regex::new(r"^\s{4}def\s+(\w+)\s*\(([^)]*)\)\s*(?:->.*)?:").unwrap();

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();
        let line_no = i + 1;

        // Skip empty lines and comments at top level
        if trimmed.is_empty() || trimmed.starts_with('#') {
            top_level.push(PyStatement {
                text: line.to_string(),
                line: line_no,
            });
            i += 1;
            continue;
        }

        // Import
        if let Some(caps) = re_from_import.captures(trimmed) {
            let module = caps[1].to_string();
            let names_str = caps[2].trim();
            let names: Vec<String> = names_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            imports.push(PyImport {
                kind: ImportKind::FromImport,
                module,
                names,
                alias: None,
                line: line_no,
            });
            i += 1;
            continue;
        }

        if let Some(caps) = re_import.captures(trimmed) {
            let module = caps[1].to_string();
            let alias = caps.get(2).map(|m| m.as_str().to_string());
            imports.push(PyImport {
                kind: ImportKind::Import,
                module,
                names: vec![],
                alias,
                line: line_no,
            });
            i += 1;
            continue;
        }

        // Class definition
        if let Some(caps) = re_class.captures(trimmed) {
            let name = caps[1].to_string();
            let bases: Vec<String> = caps[2]
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let (methods, body_lines, end) = parse_class_body(&lines, i + 1);
            classes.push(PyClass {
                name,
                bases,
                methods,
                body_lines,
                line: line_no,
            });
            i = end;
            continue;
        }

        // Top-level function
        if let Some(caps) = re_def.captures(trimmed) {
            let name = caps[1].to_string();
            let args: Vec<String> = caps[2]
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let (body, end) = parse_indented_block(&lines, i + 1, 4);
            functions.push(PyFunction {
                name,
                args,
                body,
                line: line_no,
            });
            i = end;
            continue;
        }

        // Handle method-like indentation at top level (shouldn't happen, but catch it)
        if re_method.is_match(line) {
            // Skip, it's part of something we missed
            i += 1;
            continue;
        }

        // Top-level statement
        top_level.push(PyStatement {
            text: line.to_string(),
            line: line_no,
        });
        i += 1;
    }

    PyFile {
        imports,
        classes,
        functions,
        top_level,
    }
}

/// Parse methods within a class body (indented by 4 spaces).
fn parse_class_body(
    lines: &[&str],
    start: usize,
) -> (Vec<PyMethod>, Vec<String>, usize) {
    let re_method = Regex::new(r"^\s{4}def\s+(\w+)\s*\(([^)]*)\)\s*(?:->.*)?:").unwrap();
    let mut methods = Vec::new();
    let mut body_lines = Vec::new();
    let mut i = start;

    while i < lines.len() {
        let line = lines[i];

        // End of class: non-empty, non-comment line with no indentation
        if !line.is_empty() && !line.trim().is_empty() && !line.starts_with(' ') && !line.starts_with('\t') {
            break;
        }

        if let Some(caps) = re_method.captures(line) {
            let name = caps[1].to_string();
            let args: Vec<String> = caps[2]
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let (body, end) = parse_indented_block(lines, i + 1, 8);
            methods.push(PyMethod {
                name,
                args,
                body,
                line: i + 1,
            });
            i = end;
        } else {
            body_lines.push(line.to_string());
            i += 1;
        }
    }

    (methods, body_lines, i)
}

/// Parse an indented block starting at `start`, collecting lines indented
/// at least `min_indent` spaces.
fn parse_indented_block(lines: &[&str], start: usize, min_indent: usize) -> (Vec<String>, usize) {
    let mut body = Vec::new();
    let mut i = start;

    while i < lines.len() {
        let line = lines[i];

        // Empty lines are part of the block
        if line.trim().is_empty() {
            body.push(line.to_string());
            i += 1;
            continue;
        }

        // Count leading spaces
        let indent = line.len() - line.trim_start().len();
        if indent >= min_indent {
            body.push(line.to_string());
            i += 1;
        } else {
            break;
        }
    }

    // Trim trailing empty lines
    while body.last().map_or(false, |l| l.trim().is_empty()) {
        body.pop();
    }

    (body, i)
}

/// Check if a class is an nn.Module subclass.
pub fn is_nn_module(class: &PyClass) -> bool {
    class.bases.iter().any(|b| {
        b == "nn.Module"
            || b == "torch.nn.Module"
            || b == "Module"
    })
}

/// Extract the __init__ method from a class.
pub fn get_init_method(class: &PyClass) -> Option<&PyMethod> {
    class.methods.iter().find(|m| m.name == "__init__")
}

/// Extract the forward method from a class.
pub fn get_forward_method(class: &PyClass) -> Option<&PyMethod> {
    class.methods.iter().find(|m| m.name == "forward")
}

/// Detect what torch ecosystem libraries are imported.
pub struct ImportAnalysis {
    pub uses_torch: bool,
    pub uses_nn: bool,
    pub uses_optim: bool,
    pub uses_data: bool,
    pub uses_functional: bool,
    pub torchvision_imports: Vec<PyImport>,
    pub unsupported_imports: Vec<PyImport>,
}

pub fn analyze_imports(imports: &[PyImport]) -> ImportAnalysis {
    let mut analysis = ImportAnalysis {
        uses_torch: false,
        uses_nn: false,
        uses_optim: false,
        uses_data: false,
        uses_functional: false,
        torchvision_imports: Vec::new(),
        unsupported_imports: Vec::new(),
    };

    for imp in imports {
        let m = &imp.module;
        if m == "torch" || m.starts_with("torch.") {
            analysis.uses_torch = true;
        }
        if m == "torch.nn" || m.starts_with("torch.nn.") {
            analysis.uses_nn = true;
        }
        if m.contains("nn.functional") {
            analysis.uses_functional = true;
        }
        if m == "torch.optim" || m.starts_with("torch.optim.") {
            analysis.uses_optim = true;
        }
        if m == "torch.utils.data" || m.starts_with("torch.utils.data.") {
            analysis.uses_data = true;
        }
        if m.starts_with("torchvision") || m.starts_with("torchaudio") || m.starts_with("torchtext") {
            analysis.torchvision_imports.push(imp.clone());
        }
        if m.contains("torch.jit")
            || m.contains("torch.onnx")
            || m.contains("torch.distributed")
        {
            analysis.unsupported_imports.push(imp.clone());
        }
    }

    analysis
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_simple() {
        let src = r#"
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x)

x = torch.randn(1, 784)
"#;
        let file = analyze(src);
        assert_eq!(file.imports.len(), 2);
        assert_eq!(file.classes.len(), 1);
        assert_eq!(file.classes[0].name, "Net");
        assert!(is_nn_module(&file.classes[0]));
        assert_eq!(file.classes[0].methods.len(), 2);
        assert!(get_init_method(&file.classes[0]).is_some());
        assert!(get_forward_method(&file.classes[0]).is_some());
    }

    #[test]
    fn test_analyze_imports() {
        let src = r#"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
"#;
        let file = analyze(src);
        let analysis = analyze_imports(&file.imports);
        assert!(analysis.uses_torch);
        assert!(analysis.uses_nn);
        assert!(analysis.uses_optim);
        assert!(analysis.uses_data);
        assert_eq!(analysis.torchvision_imports.len(), 1);
    }
}
