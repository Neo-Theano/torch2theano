use regex::Regex;
use std::fmt;

/// A single recorded change applied to a source file.
#[derive(Debug, Clone)]
pub struct Change {
    pub line: usize,
    pub kind: ChangeKind,
    pub description: String,
    #[allow(dead_code)]
    pub original: String,
    #[allow(dead_code)]
    pub replacement: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeKind {
    Import,
    Namespace,
    Dependency,
    Warning,
}

impl fmt::Display for ChangeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChangeKind::Import => write!(f, "import"),
            ChangeKind::Namespace => write!(f, "namespace"),
            ChangeKind::Dependency => write!(f, "dependency"),
            ChangeKind::Warning => write!(f, "warning"),
        }
    }
}

/// Outcome of transforming a single source file.
pub struct TransformResult {
    pub output: String,
    pub changes: Vec<Change>,
}

/// Rule table entry: (regex_pattern, replacement_template, description, kind)
struct Rule {
    pattern: Regex,
    replacement: &'static str,
    description: &'static str,
    kind: ChangeKind,
}

impl Rule {
    fn new(pat: &str, repl: &'static str, desc: &'static str, kind: ChangeKind) -> Self {
        Self {
            pattern: Regex::new(pat).expect("invalid rule regex"),
            replacement: repl,
            description: desc,
            kind,
        }
    }
}

/// Build the ordered list of transformation rules.
fn build_rules() -> Vec<Rule> {
    let mut rules = Vec::new();

    // ── Import transforms ────────────────────────────────────────────
    // Order matters: more specific patterns first.

    rules.push(Rule::new(
        r"^(\s*)from\s+torch\.utils\.data\s+import\b",
        "${1}from theano.data import",
        "torch.utils.data import -> theano.data",
        ChangeKind::Import,
    ));
    rules.push(Rule::new(
        r"^(\s*)from\s+torch\.nn\.functional\s+import\b",
        "${1}from theano.nn.functional import",
        "torch.nn.functional import -> theano.nn.functional",
        ChangeKind::Import,
    ));
    rules.push(Rule::new(
        r"^(\s*)from\s+torch\.optim\s+import\b",
        "${1}from theano.optim import",
        "torch.optim import -> theano.optim",
        ChangeKind::Import,
    ));
    rules.push(Rule::new(
        r"^(\s*)from\s+torch\.nn\s+import\b",
        "${1}from theano.nn import",
        "torch.nn import -> theano.nn",
        ChangeKind::Import,
    ));
    rules.push(Rule::new(
        r"^(\s*)from\s+torch\s+import\b",
        "${1}from theano import",
        "torch import -> theano",
        ChangeKind::Import,
    ));
    rules.push(Rule::new(
        r"^(\s*)import\s+torch\.utils\.data\b",
        "${1}import theano.data",
        "import torch.utils.data -> theano.data",
        ChangeKind::Import,
    ));
    rules.push(Rule::new(
        r"^(\s*)import\s+torch\.nn\.functional\b",
        "${1}import theano.nn.functional",
        "import torch.nn.functional -> theano.nn.functional",
        ChangeKind::Import,
    ));
    rules.push(Rule::new(
        r"^(\s*)import\s+torch\.optim\b",
        "${1}import theano.optim",
        "import torch.optim -> theano.optim",
        ChangeKind::Import,
    ));
    rules.push(Rule::new(
        r"^(\s*)import\s+torch\.nn\b",
        "${1}import theano.nn",
        "import torch.nn -> theano.nn",
        ChangeKind::Import,
    ));
    rules.push(Rule::new(
        r"^(\s*)import\s+torch\b",
        "${1}import theano",
        "import torch -> theano",
        ChangeKind::Import,
    ));

    // ── Qualified‐name / namespace transforms ────────────────────────
    // Replace `torch.` prefixed identifiers in expressions.
    // We use \b (word boundary) so we don't match inside another
    // identifier (e.g. `my_torch.zeros` won't match because there is
    // no word boundary between `_` and `t`).  Note: `_` IS a word
    // character in regex, so `\btorch` already excludes `_torch`.

    rules.push(Rule::new(
        r"\btorch\.utils\.data\.",
        "theano.data.",
        "torch.utils.data.X -> theano.data.X",
        ChangeKind::Namespace,
    ));
    rules.push(Rule::new(
        r"\btorch\.nn\.functional\.",
        "theano.nn.functional.",
        "torch.nn.functional.X -> theano.nn.functional.X",
        ChangeKind::Namespace,
    ));
    rules.push(Rule::new(
        r"\btorch\.nn\.",
        "theano.nn.",
        "torch.nn.X -> theano.nn.X",
        ChangeKind::Namespace,
    ));
    rules.push(Rule::new(
        r"\btorch\.optim\.",
        "theano.optim.",
        "torch.optim.X -> theano.optim.X",
        ChangeKind::Namespace,
    ));
    rules.push(Rule::new(
        r"\btorch\.cuda\.",
        "theano.cuda.",
        "torch.cuda.X -> theano.cuda.X",
        ChangeKind::Namespace,
    ));

    // Generic torch.X -> theano.X (must come after the more-specific rules above).
    rules.push(Rule::new(
        r"\btorch\.",
        "theano.",
        "torch.X -> theano.X",
        ChangeKind::Namespace,
    ));

    rules
}

/// Build rules that detect constructs requiring manual attention.
fn build_warning_rules() -> Vec<Rule> {
    vec![
        Rule::new(
            r"\btorchvision\b",
            "",
            "torchvision has no direct Neo Theano equivalent – manual porting required",
            ChangeKind::Warning,
        ),
        Rule::new(
            r"\btorchaudio\b",
            "",
            "torchaudio has no direct Neo Theano equivalent – manual porting required",
            ChangeKind::Warning,
        ),
        Rule::new(
            r"\btorchtext\b",
            "",
            "torchtext has no direct Neo Theano equivalent – manual porting required",
            ChangeKind::Warning,
        ),
        Rule::new(
            r"\btorch\.jit\b",
            "",
            "torch.jit – Neo Theano uses a different JIT system",
            ChangeKind::Warning,
        ),
        Rule::new(
            r"\btorch\.onnx\b",
            "",
            "torch.onnx export – review Neo Theano serialization options",
            ChangeKind::Warning,
        ),
        Rule::new(
            r"\btorch\.distributed\b",
            "",
            "torch.distributed – Neo Theano distributed API may differ",
            ChangeKind::Warning,
        ),
        Rule::new(
            r"\btorch\.multiprocessing\b",
            "",
            "torch.multiprocessing – review Neo Theano multiprocessing support",
            ChangeKind::Warning,
        ),
        Rule::new(
            r"\bload_state_dict\b",
            "",
            "load_state_dict – verify state dict key compatibility",
            ChangeKind::Warning,
        ),
        Rule::new(
            r"\.cuda\(\)",
            "",
            ".cuda() – consider using .to(device) for backend portability",
            ChangeKind::Warning,
        ),
    ]
}

/// Transform Python source code from PyTorch to Neo Theano.
pub fn transform_python(source: &str) -> TransformResult {
    let rules = build_rules();
    let warning_rules = build_warning_rules();

    let mut changes: Vec<Change> = Vec::new();
    let mut output_lines: Vec<String> = Vec::new();

    let mut in_multiline_str = false;
    let mut ml_delim: &str = "";

    for (line_idx, raw_line) in source.lines().enumerate() {
        let line_no = line_idx + 1;

        // ── Track multiline strings (""" / ''') ──────────────────
        if in_multiline_str {
            if raw_line.contains(ml_delim) {
                in_multiline_str = false;
            }
            output_lines.push(raw_line.to_string());
            continue;
        }

        // Check if a multiline string starts on this line.
        if starts_multiline(raw_line) {
            let delim = if raw_line.contains(r#"""""#) {
                r#"""""#
            } else {
                "'''"
            };
            // If there's no closing delimiter on the same line, mark multiline.
            let first = raw_line.find(delim).unwrap();
            if raw_line[first + 3..].find(delim).is_none() {
                in_multiline_str = true;
                ml_delim = delim;
            }
            output_lines.push(raw_line.to_string());
            continue;
        }

        // ── Skip pure comment lines ──────────────────────────────
        let trimmed = raw_line.trim();
        if trimmed.starts_with('#') {
            output_lines.push(raw_line.to_string());
            continue;
        }

        // ── Separate code from inline comment ────────────────────
        let (code_part, comment_part) = split_comment(raw_line);

        // ── Apply warning rules (detect, don't modify) ──────────
        for wr in &warning_rules {
            if wr.pattern.is_match(code_part) {
                changes.push(Change {
                    line: line_no,
                    kind: ChangeKind::Warning,
                    description: wr.description.to_string(),
                    original: raw_line.to_string(),
                    replacement: String::new(),
                });
            }
        }

        // ── Apply transform rules ────────────────────────────────
        let mut transformed = code_part.to_string();
        for rule in &rules {
            if rule.pattern.is_match(&transformed) {
                let before = transformed.clone();
                transformed = rule.pattern.replace_all(&transformed, rule.replacement).to_string();
                if before != transformed {
                    changes.push(Change {
                        line: line_no,
                        kind: rule.kind,
                        description: rule.description.to_string(),
                        original: before,
                        replacement: transformed.clone(),
                    });
                }
            }
        }

        // Reassemble with inline comment.
        if comment_part.is_empty() {
            output_lines.push(transformed);
        } else {
            output_lines.push(format!("{}  {}", transformed, comment_part));
        }
    }

    // Preserve final newline if original had one.
    let mut output = output_lines.join("\n");
    if source.ends_with('\n') {
        output.push('\n');
    }

    TransformResult { output, changes }
}

/// Transform dependency files (requirements.txt, setup.py, pyproject.toml).
pub fn transform_dependency_file(source: &str, filename: &str) -> TransformResult {
    let mut changes: Vec<Change> = Vec::new();
    let mut output_lines: Vec<String> = Vec::new();

    let torch_dep = Regex::new(r"(?m)^(\s*)torch\b([>=<!\s].*)$").unwrap();
    let torchvision_dep = Regex::new(r"(?m)^(\s*)torchvision\b").unwrap();
    let torchaudio_dep = Regex::new(r"(?m)^(\s*)torchaudio\b").unwrap();
    let torchtext_dep = Regex::new(r"(?m)^(\s*)torchtext\b").unwrap();
    let torch_quoted = Regex::new(r#"["']torch([>=<!\s][^"']*)["']"#).unwrap();
    let torchvision_quoted = Regex::new(r#"["']torchvision([>=<!\s][^"']*)["']"#).unwrap();
    let torchaudio_quoted = Regex::new(r#"["']torchaudio([>=<!\s][^"']*)["']"#).unwrap();

    for (line_idx, raw_line) in source.lines().enumerate() {
        let line_no = line_idx + 1;
        let mut line = raw_line.to_string();

        // Plain dependency lines (requirements.txt style)
        if torch_dep.is_match(&line) {
            let before = line.clone();
            line = torch_dep.replace(&line, "${1}theano${2}").to_string();
            changes.push(Change {
                line: line_no,
                kind: ChangeKind::Dependency,
                description: format!("torch dependency -> theano in {}", filename),
                original: before,
                replacement: line.clone(),
            });
        }

        // Quoted dependencies (setup.py / pyproject.toml style)
        if torch_quoted.is_match(&line) {
            let before = line.clone();
            line = torch_quoted.replace_all(&line, "\"theano${1}\"").to_string();
            changes.push(Change {
                line: line_no,
                kind: ChangeKind::Dependency,
                description: format!("torch dependency -> theano in {}", filename),
                original: before,
                replacement: line.clone(),
            });
        }
        if torchvision_quoted.is_match(&line) {
            let before = line.clone();
            line = torchvision_quoted
                .replace_all(&line, "# \"torchvision${1}\"  # TODO: no Neo Theano equivalent")
                .to_string();
            changes.push(Change {
                line: line_no,
                kind: ChangeKind::Warning,
                description: "torchvision dependency commented out".into(),
                original: before,
                replacement: line.clone(),
            });
        }
        if torchaudio_quoted.is_match(&line) {
            let before = line.clone();
            line = torchaudio_quoted
                .replace_all(&line, "# \"torchaudio${1}\"  # TODO: no Neo Theano equivalent")
                .to_string();
            changes.push(Change {
                line: line_no,
                kind: ChangeKind::Warning,
                description: "torchaudio dependency commented out".into(),
                original: before,
                replacement: line.clone(),
            });
        }

        // Unquoted ecosystem deps
        if torchvision_dep.is_match(&line) && !line.contains('"') && !line.contains('\'') {
            let before = line.clone();
            line = format!("# {}  # TODO: no Neo Theano equivalent", line.trim());
            changes.push(Change {
                line: line_no,
                kind: ChangeKind::Warning,
                description: "torchvision dependency commented out".into(),
                original: before,
                replacement: line.clone(),
            });
        }
        if torchaudio_dep.is_match(&line) && !line.contains('"') && !line.contains('\'') {
            let before = line.clone();
            line = format!("# {}  # TODO: no Neo Theano equivalent", line.trim());
            changes.push(Change {
                line: line_no,
                kind: ChangeKind::Warning,
                description: "torchaudio dependency commented out".into(),
                original: before,
                replacement: line.clone(),
            });
        }
        if torchtext_dep.is_match(&line) && !line.contains('"') && !line.contains('\'') {
            let before = line.clone();
            line = format!("# {}  # TODO: no Neo Theano equivalent", line.trim());
            changes.push(Change {
                line: line_no,
                kind: ChangeKind::Warning,
                description: "torchtext dependency commented out".into(),
                original: before,
                replacement: line.clone(),
            });
        }

        output_lines.push(line);
    }

    let mut output = output_lines.join("\n");
    if source.ends_with('\n') {
        output.push('\n');
    }
    TransformResult { output, changes }
}

// ── helpers ──────────────────────────────────────────────────────────

/// Checks if a line contains an opening multiline string delimiter that
/// is not closed on the same line.
fn starts_multiline(line: &str) -> bool {
    let trimmed = line.trim();
    // Skip lines that are pure comments.
    if trimmed.starts_with('#') {
        return false;
    }
    for delim in [r#"""""#, "'''"] {
        if let Some(first) = line.find(delim) {
            // If there is only one occurrence, it's an opening delimiter.
            if line[first + 3..].find(delim).is_none() {
                return true;
            }
        }
    }
    false
}

/// Split a line into (code, comment).  Handles `#` inside strings naively
/// (a full parser would be needed for perfect accuracy, but this covers
/// the vast majority of real-world Python).
fn split_comment(line: &str) -> (&str, &str) {
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
                return (line[..i].trim_end(), &line[i..]);
            }
            '\\' => {
                i += 1; // skip escaped character
            }
            _ => {}
        }
        i += 1;
    }
    (line, "")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_import_rewrite() {
        let src = "import torch\nimport torch.nn as nn\n";
        let r = transform_python(src);
        assert!(r.output.contains("import theano\n"));
        assert!(r.output.contains("import theano.nn as nn\n"));
        assert!(!r.changes.is_empty());
    }

    #[test]
    fn test_from_import_rewrite() {
        let src = "from torch import Tensor\nfrom torch.nn import Linear, Conv2d\n";
        let r = transform_python(src);
        assert!(r.output.contains("from theano import Tensor\n"));
        assert!(r.output.contains("from theano.nn import Linear, Conv2d\n"));
    }

    #[test]
    fn test_namespace_rewrite() {
        let src = "x = torch.zeros(3, 4)\ny = torch.randn(2, 2)\n";
        let r = transform_python(src);
        assert!(r.output.contains("theano.zeros"));
        assert!(r.output.contains("theano.randn"));
    }

    #[test]
    fn test_no_transform_in_comment() {
        let src = "# import torch\n";
        let r = transform_python(src);
        assert_eq!(r.output, "# import torch\n");
        assert!(r.changes.is_empty());
    }

    #[test]
    fn test_warning_for_torchvision() {
        let src = "import torchvision\n";
        let r = transform_python(src);
        let warnings: Vec<_> = r.changes.iter().filter(|c| c.kind == ChangeKind::Warning).collect();
        assert!(!warnings.is_empty());
    }

    #[test]
    fn test_utils_data_rewrite() {
        let src = "from torch.utils.data import DataLoader, Dataset\n";
        let r = transform_python(src);
        assert!(r.output.contains("from theano.data import DataLoader, Dataset\n"));
    }

    #[test]
    fn test_dependency_rewrite() {
        let src = "torch>=2.0\ntorchvision>=0.15\nnumpy\n";
        let r = transform_dependency_file(src, "requirements.txt");
        assert!(r.output.contains("theano>=2.0"));
        assert!(r.output.contains("# torchvision"));
    }

    #[test]
    fn test_multiline_string_skipped() {
        let src = r#"msg = """
import torch
torch.zeros(3)
"""
x = torch.ones(2)
"#;
        let r = transform_python(src);
        // The code outside the multiline string should be transformed.
        assert!(r.output.contains("theano.ones"));
        // The content inside the multiline string should NOT be transformed.
        assert!(r.output.contains("import torch\ntorch.zeros(3)"));
    }
}
