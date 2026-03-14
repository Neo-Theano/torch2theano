// Converted from PyTorch Python to Neo Theano Rust by torch2theano.
//
// NOTE: Some PyTorch features (datasets, transforms, image saving)
// have no Neo Theano equivalent yet. Dataset loading is stubbed with
// random tensors as placeholder data.

use theano::prelude::*;
use theano::nn::*;
use theano::optim::{Adam, Optimizer};
use theano_serialize::{save_state_dict, load_state_dict};
use clap::Parser;
use rand::Rng;

/// Generator neural network module.
struct Generator {
    main: Sequential,
}

impl Generator {
    fn new(nz: usize, ngf: usize, nc: usize, _ngpu: usize) -> Self {
        let main = Sequential::new(vec![])
            .add(Conv2d::with_options(nz, ngf * 8, (4, 4), (1, 1), (0, 0), false))
            .add(BatchNorm1d::new(ngf * 8))
            .add(ReLU)
            .add(Conv2d::with_options(ngf * 8, ngf * 4, (4, 4), (2, 2), (1, 1), false))
            .add(BatchNorm1d::new(ngf * 4))
            .add(ReLU)
            .add(Conv2d::with_options(ngf * 4, ngf * 2, (4, 4), (2, 2), (1, 1), false))
            .add(BatchNorm1d::new(ngf * 2))
            .add(ReLU)
            .add(Conv2d::with_options(ngf * 2, ngf, (4, 4), (2, 2), (1, 1), false))
            .add(BatchNorm1d::new(ngf))
            .add(ReLU)
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
    fn new(nc: usize, ndf: usize, _ngpu: usize) -> Self {
        let main = Sequential::new(vec![])
            .add(Conv2d::with_options(nc, ndf, (4, 4), (2, 2), (1, 1), false))
            .add(ReLU) // TODO: LeakyReLU(0.2)
            .add(Conv2d::with_options(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), false))
            .add(BatchNorm1d::new(ndf * 2))
            .add(ReLU) // TODO: LeakyReLU(0.2)
            .add(Conv2d::with_options(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), false))
            .add(BatchNorm1d::new(ndf * 4))
            .add(ReLU) // TODO: LeakyReLU(0.2)
            .add(Conv2d::with_options(ndf * 4, ndf * 8, (4, 4), (2, 2), (1, 1), false))
            .add(BatchNorm1d::new(ndf * 8))
            .add(ReLU) // TODO: LeakyReLU(0.2)
            .add(Conv2d::with_options(ndf * 8, 1, (4, 4), (1, 1), (0, 0), false))
            .add(Sigmoid);
        Self { main }
    }
}

impl Module for Discriminator {
    fn forward(&self, input: &Variable) -> Variable {
        let x = self.main.forward(input);
        x.view(&[-1, 1]).unwrap()
    }

    fn parameters(&self) -> Vec<Variable> {
        self.main.parameters()
    }
}

/// Weight initialization placeholder.
fn weights_init(_module: &dyn Module) {
    // TODO: Apply per-layer initialization:
    // - Conv layers: normal_(weight, 0.0, 0.02)
    // - BatchNorm layers: normal_(weight, 1.0, 0.02), zeros_(bias)
}

#[derive(Parser)]
#[command(about = "DCGAN training example")]
struct Args {
    /// cifar10 | lsun | mnist | imagenet | folder | lfw | fake
    #[arg(long)]
    dataset: Option<String>,

    /// path to dataset
    #[arg(long)]
    dataroot: Option<String>,

    /// number of data loading workers
    #[arg(long, default_value_t = 2)]
    workers: usize,

    /// input batch size
    #[arg(long, default_value_t = 64)]
    batchsize: usize,

    /// the height / width of the input image to network
    #[arg(long, default_value_t = 64)]
    imagesize: usize,

    /// size of the latent z vector
    #[arg(long, default_value_t = 100)]
    nz: usize,

    /// number of generator filters
    #[arg(long, default_value_t = 64)]
    ngf: usize,

    /// number of discriminator filters
    #[arg(long, default_value_t = 64)]
    ndf: usize,

    /// number of epochs to train for
    #[arg(long, default_value_t = 25)]
    niter: usize,

    /// learning rate
    #[arg(long, default_value_t = 0.0002)]
    lr: f64,

    /// beta1 for adam
    #[arg(long, default_value_t = 0.5)]
    beta1: f64,

    /// check a single training cycle works
    #[arg(long)]
    dry_run: bool,

    /// number of GPUs to use
    #[arg(long, default_value_t = 1)]
    ngpu: usize,

    /// path to netG (to continue training)
    #[arg(long, default_value = "")]
    netg: String,

    /// path to netD (to continue training)
    #[arg(long, default_value = "")]
    netd: String,

    /// folder to output images and model checkpoints
    #[arg(long, default_value = ".")]
    outf: String,

    /// manual seed
    #[arg(long)]
    manualseed: Option<usize>,

    /// comma separated list of classes for the lsun data set
    #[arg(long, default_value = "bedroom")]
    classes: String,

    /// enables accelerator
    #[arg(long)]
    accel: bool,
}

fn main() {
    let args = Args::parse();

    // Create output directory
    std::fs::create_dir_all(&args.outf).ok();

    // Set random seed
    let seed = args.manualseed.unwrap_or_else(|| {
        rand::thread_rng().gen_range(1..=10000)
    });
    println!("Random Seed: {}", seed);

    // Determine device
    let _device = if args.accel {
        // TODO: check accelerator availability
        Device::Cpu
    } else {
        Device::Cpu
    };

    // Number of channels (depends on dataset)
    let nc: usize = match args.dataset.as_deref() {
        Some("mnist") => 1,
        _ => 3,
    };

    let nz = args.nz;
    let ngf = args.ngf;
    let ndf = args.ndf;
    let ngpu = args.ngpu;

    // Create Generator
    let net_g = Generator::new(nz, ngf, nc, ngpu);
    weights_init(&net_g);
    if !args.netg.is_empty() {
        let bytes = std::fs::read(&args.netg).expect("failed to read generator checkpoint");
        let _state = load_state_dict(&bytes).expect("failed to load generator state dict");
        // TODO: apply _state to net_g
    }
    println!("Generator created with {} parameters", net_g.num_parameters());

    // Create Discriminator
    let net_d = Discriminator::new(nc, ndf, ngpu);
    weights_init(&net_d);
    if !args.netd.is_empty() {
        let bytes = std::fs::read(&args.netd).expect("failed to read discriminator checkpoint");
        let _state = load_state_dict(&bytes).expect("failed to load discriminator state dict");
        // TODO: apply _state to net_d
    }
    println!("Discriminator created with {} parameters", net_d.num_parameters());

    let criterion = BCELoss::new();
    let fixed_noise = Variable::new(Tensor::randn(&[args.batchsize, nz, 1, 1]));
    let real_label: f64 = 1.0;
    let fake_label: f64 = 0.0;

    // Setup optimizers
    let mut optimizer_d = Adam::new(net_d.parameters(), args.lr)
        .betas(args.beta1, 0.999);
    let mut optimizer_g = Adam::new(net_g.parameters(), args.lr)
        .betas(args.beta1, 0.999);

    let niter = if args.dry_run { 1 } else { args.niter };

    // TODO: Dataset/DataLoader not yet available in Neo Theano.
    // Using random tensors as placeholder data.
    for epoch in 0..niter {
        let num_batches: usize = 1; // placeholder until DataLoader is available
        for i in 0..num_batches {
            let batch_size = args.batchsize;

            // ============================
            // (1) Update D: maximize log(D(x)) + log(1 - D(G(z)))
            // ============================
            optimizer_d.zero_grad();

            // Train with real (placeholder random data)
            let real_cpu = Variable::new(
                Tensor::randn(&[batch_size, nc, args.imagesize, args.imagesize]),
            );
            let label = Variable::new(Tensor::full(&[batch_size], real_label));
            let output = net_d.forward(&real_cpu);
            let err_d_real = criterion.forward(&output, &label);
            err_d_real.backward();

            // Train with fake
            let noise = Variable::new(Tensor::randn(&[batch_size, nz, 1, 1]));
            let fake = net_g.forward(&noise);
            let label = Variable::new(Tensor::full(&[batch_size], fake_label));
            let output = net_d.forward(&fake.detach());
            let err_d_fake = criterion.forward(&output, &label);
            err_d_fake.backward();

            optimizer_d.step();

            // ============================
            // (2) Update G: maximize log(D(G(z)))
            // ============================
            optimizer_g.zero_grad();
            let label = Variable::new(Tensor::full(&[batch_size], real_label));
            let output = net_d.forward(&fake);
            let err_g = criterion.forward(&output, &label);
            err_g.backward();
            optimizer_g.step();

            println!(
                "[{}/{}][{}/{}] Training step complete",
                epoch, niter, i, num_batches,
            );

            if i % 100 == 0 {
                let _fake = net_g.forward(&fixed_noise);
                // TODO: save_image not available (torchvision equivalent needed)
            }

            if args.dry_run {
                break;
            }
        }

        // Checkpointing
        let bytes = save_state_dict(&net_g.state_dict());
        std::fs::write(
            format!("{}/netG_epoch_{}.pth", args.outf, epoch),
            bytes,
        )
        .expect("failed to save generator checkpoint");

        let bytes = save_state_dict(&net_d.state_dict());
        std::fs::write(
            format!("{}/netD_epoch_{}.pth", args.outf, epoch),
            bytes,
        )
        .expect("failed to save discriminator checkpoint");
    }
}
