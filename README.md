# torch2theano

Convert PyTorch Python source code to [Neo Theano](https://github.com/Neo-Theano/theano/).

## Why this exists

**Ecosystem is the moat.** Every ML paper published in the last 5 years ships with PyTorch code. Hugging Face has 200k+ pretrained models. torchvision, torchaudio, torchtext provide ready-made data pipelines. This ecosystem took a decade to build and cannot be replicated by any new framework on technical merit alone.

**Community and hiring.** Practically every ML engineer knows PyTorch. Practically none know Rust autodiff. For organizations, this means PyTorch projects can hire from a deep talent pool, find answers on StackOverflow, and get support from a massive community.

`torch2theano` bridges this gap by automatically converting PyTorch repositories to Neo Theano, letting you adopt Neo Theano without rewriting everything from scratch.

## Install

```bash
cargo install --path .
```

Or build from source:

```bash
cargo build --release
# binary is at ./target/release/torch2theano
```

## Usage

```bash
# Convert a PyTorch repo to Neo Theano (output written to <dir>_theano/)
torch2theano /path/to/pytorch-repo

# Specify a custom output directory
torch2theano /path/to/pytorch-repo --output /path/to/output

# Dry run — see what would change without writing files
torch2theano /path/to/pytorch-repo --dry-run

# Verbose — show per-file change details
torch2theano /path/to/pytorch-repo --verbose
```

## What it converts

| PyTorch | Neo Theano | Status |
|---------|-----------|--------|
| `import torch` | `import theano` | automatic |
| `import torch.nn as nn` | `import theano.nn as nn` | automatic |
| `import torch.optim` | `import theano.optim` | automatic |
| `from torch.utils.data import DataLoader` | `from theano.data import DataLoader` | automatic |
| `torch.tensor(...)` | `theano.tensor(...)` | automatic |
| `torch.zeros(...)`, `torch.ones(...)`, `torch.randn(...)` | `theano.zeros(...)`, `theano.ones(...)`, `theano.randn(...)` | automatic |
| `torch.cuda.is_available()` | `theano.cuda.is_available()` | automatic |
| `torch.no_grad()` | `theano.no_grad()` | automatic |
| `torch.save(...)` / `torch.load(...)` | `theano.save(...)` / `theano.load(...)` | automatic |
| `nn.Module`, `nn.Linear`, `nn.Conv2d`, etc. | Same API | automatic |
| `nn.CrossEntropyLoss`, `nn.MSELoss`, etc. | Same API | automatic |
| `optim.Adam`, `optim.SGD`, etc. | Same API | automatic |
| `requirements.txt` torch dependency | Rewritten to theano | automatic |
| `torchvision`, `torchaudio`, `torchtext` | — | warning (manual) |
| `torch.jit`, `torch.onnx` | — | warning (manual) |
| `torch.distributed` | — | warning (manual) |

### What it does NOT convert

- **torchvision / torchaudio / torchtext**: These ecosystem libraries have no direct Neo Theano equivalent. The tool flags them as warnings so you know what needs manual attention.
- **Custom CUDA extensions**: C++/CUDA kernels need to be ported to Neo Theano's backend system.
- **torch.jit / TorchScript**: Neo Theano has its own JIT system.
- **Content inside strings and comments**: Intentionally left untouched.

## Example

Input (`model.py`):
```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x)

x = torch.randn(1, 784)
model = Net()
print(model(x))
```

Output:
```python
import theano
import theano.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x)

x = theano.randn(1, 784)
model = Net()
print(model(x))
```

## Conversion report

After conversion, `torch2theano` prints a summary:

```
─── Conversion Report ───
  files scanned:     42
  files transformed: 18
  files copied:      24
  files skipped:     0
  total changes:     87
  warnings:          3

  Changes by category:
    imports        34
    namespaces     52
    dependencies   1

  Warnings (require manual review):
    requirements.txt:2 torchvision dependency commented out
    train.py:5 torchvision has no direct Neo Theano equivalent
    utils.py:12 torch.jit – Neo Theano uses a different JIT system
```

## License

MIT
