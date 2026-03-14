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

## Example: DCGAN

A full example is included in [`example/`](example/) using PyTorch's [DCGAN](https://github.com/pytorch/examples/blob/main/dcgan/main.py) implementation. The converter takes the Python source and produces a Rust project with `Cargo.toml` and `src/main.rs`.

### Input: PyTorch Python ([`example/dcgan/main.py`](example/dcgan/main.py))

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # ...
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # ...
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(niter):
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    fake = netG(noise)
    output = netD(fake.detach())
    errD_fake = criterion(output, label)
    errD_fake.backward()
    optimizerD.step()

    torch.save(netG.state_dict(), 'netG_epoch_%d.pth' % epoch)
```

### Output: Neo Theano Rust project ([`example/dcgan_theano/`](example/dcgan_theano/))

`Cargo.toml`:
```toml
[package]
name = "dcgan"
version = "0.1.0"
edition = "2021"

[dependencies]
theano = { git = "https://github.com/Neo-Theano/theano.git", features = ["full"] }
clap = { version = "4", features = ["derive"] }
rand = "0.8"
```

`src/main.rs` (excerpts):
```rust
use theano::prelude::*;
use theano::nn::*;
use theano::optim::{Adam, SGD, Optimizer};
use theano_types::Device;
use clap::Parser;

/// Generator neural network module.
struct Generator {
    main: Sequential,
}

impl Generator {
    fn new(ngpu: usize) -> Self {
        let main = Sequential::new(vec![])
            .add(Conv2d::with_options(nz, ngf * 8, (4, 4), (1, 1), (0, 0), false))
            .add(BatchNorm1d::new(ngf * 8))
            .add(ReLU)
            .add(Conv2d::with_options(ngf * 8, ngf * 4, (4, 4), (2, 2), (1, 1), false))
            .add(BatchNorm1d::new(ngf * 4))
            .add(ReLU)
            // ... (3 more ConvTranspose blocks)
            .add(Conv2d::with_options(ngf, nc, (4, 4), (2, 2), (1, 1), false))
            .add(Tanh);
        Self { main }
    }
}

impl Module for Generator {
    fn forward(&self, input: &Variable) -> Variable {
        self.main.forward(input)
    }
    fn parameters(&self) -> Vec<Variable> {
        self.main.parameters()
    }
}

/// Discriminator neural network module.
struct Discriminator {
    main: Sequential,
}

impl Discriminator {
    fn new(ngpu: usize) -> Self {
        let main = Sequential::new(vec![])
            .add(Conv2d::with_options(nc, ndf, (4, 4), (2, 2), (1, 1), false))
            .add(ReLU /* TODO: LeakyReLU(0.2) */)
            .add(Conv2d::with_options(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), false))
            .add(BatchNorm1d::new(ndf * 2))
            .add(ReLU /* TODO: LeakyReLU(0.2) */)
            // ... (2 more Conv blocks)
            .add(Conv2d::with_options(ndf * 8, 1, (4, 4), (1, 1), (0, 0), false))
            .add(Sigmoid);
        Self { main }
    }
}

impl Module for Discriminator {
    fn forward(&self, input: &Variable) -> Variable {
        let x = self.main.forward(input);
        x.reshape(&[-1, 1]).unwrap()
    }
    fn parameters(&self) -> Vec<Variable> {
        self.main.parameters()
    }
}

// argparse -> clap Args struct (auto-generated)
#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 64)]
    batchsize: usize,
    #[arg(long, default_value_t = 100)]
    nz: usize,
    // ... all CLI args with types and defaults
}

fn main() {
    let args = Args::parse();

    let netG = Generator::new(ngpu);
    let netD = Discriminator::new(ngpu);
    let criterion = BCELoss::new();

    let mut optimizerD = Adam::new(netD.parameters(), opt.lr).betas(opt.beta1, 0.999);
    let mut optimizerG = Adam::new(netG.parameters(), opt.lr).betas(opt.beta1, 0.999);

    for epoch in 0..opt.niter {
        // training loop with backward() and step()
        let noise = Variable::new(Tensor::randn(&[batch_size, nz, 1, 1]));
        let fake = netG(noise);
        // ...
    }
}
```

The full files are in [`example/dcgan/`](example/dcgan/) (input) and [`example/dcgan_theano/`](example/dcgan_theano/) (output).

## Conversion report

After conversion, `torch2theano` prints a summary:

```
─── Conversion Report ───
  files scanned:     1
  files transformed: 1
  files copied:      0
  files skipped:     0

  Converted:
    main.py -> dcgan_theano/src/main.rs
    (generated) -> dcgan_theano/Cargo.toml

  Warnings (require manual review):
    main.py:11 torchvision has no direct Neo Theano equivalent – manual porting required
    main.py:12 torchvision has no direct Neo Theano equivalent – manual porting required
    main.py:13 torchvision has no direct Neo Theano equivalent – manual porting required
    main.py:165 load_state_dict – verify state dict key compatibility
    main.py:206 load_state_dict – verify state dict key compatibility
```

## License

MIT
