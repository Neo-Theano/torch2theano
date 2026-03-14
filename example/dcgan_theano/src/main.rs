// Converted from PyTorch Python to Neo Theano Rust by torch2theano.
//
// Manual attention required:
//   - torchvision.datasets has no Neo Theano equivalent
//   - torchvision.transforms has no Neo Theano equivalent
//   - torchvision.utils has no Neo Theano equivalent
//

use theano::prelude::*;
use theano::nn::*;
use theano::optim::{Adam, Optimizer};
// use theano::data::{DataLoader, Dataset};
use theano_serialize::{save_state_dict, load_state_dict};
use clap::Parser;
use rand::Rng;

/// Generator neural network module.
struct Generator {
    main: Sequential,
}

impl Generator {
    fn new(nz: usize, ngf: usize, nc: usize, ngpu: usize) -> Self {
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
        Self {
            main,
        }
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
    fn new(nc: usize, ndf: usize, ngpu: usize) -> Self {
        let main = Sequential::new(vec![])
            .add(Conv2d::with_options(nc, ndf, (4, 4), (2, 2), (1, 1), false))
            .add(ReLU /* TODO: LeakyReLU(0.2) */)
            .add(Conv2d::with_options(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), false))
            .add(BatchNorm1d::new(ndf * 2))
            .add(ReLU /* TODO: LeakyReLU(0.2) */)
            .add(Conv2d::with_options(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), false))
            .add(BatchNorm1d::new(ndf * 4))
            .add(ReLU /* TODO: LeakyReLU(0.2) */)
            .add(Conv2d::with_options(ndf * 4, ndf * 8, (4, 4), (2, 2), (1, 1), false))
            .add(BatchNorm1d::new(ndf * 8))
            .add(ReLU /* TODO: LeakyReLU(0.2) */)
            .add(Conv2d::with_options(ndf * 8, 1, (4, 4), (1, 1), (0, 0), false))
            .add(Sigmoid);
        Self {
            main,
        }
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

/// weights init placeholder.
fn weights_init(_m: &dyn Module) {
    // TODO: classname = m.__class__.__name__
    // TODO: if classname.find('Conv') != -1:
    // TODO: torch.nn.init.normal_(m.weight, 0.0, 0.02)
    // TODO: elif classname.find('BatchNorm') != -1:
    // TODO: torch.nn.init.normal_(m.weight, 1.0, 0.02)
    // TODO: torch.nn.init.zeros_(m.bias)
}

#[derive(Parser)]
#[command(about = "Converted from PyTorch")]
struct Args {
    /// cifar10 | lsun | mnist |imagenet | folder | lfw | fake
    #[arg(long)]
    dataset: String,

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

    /// learning rate, default=0.0002
    #[arg(long, default_value_t = 0.0002)]
    lr: f64,

    /// beta1 for adam. default=0.5
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

    // TODO: print opt
    // TODO: try:
    // TODO: os.makedirs(args.outf)
    // TODO: except OSError:
    if args.manualseed.is_none() {
        // TODO: args.manualseed = random.randint(1, 10000)
    }
    println!("{:?} {:?}", "Random Seed: ", args.manualseed);
    // TODO: random.seed(args.manualseed)
    // TODO: torch.manual_seed(args.manualseed)
    // TODO: cudnn.benchmark = True
    if /* TODO: args.accel and torch.accelerator.is_available() */ false {
        // TODO: device = torch.accelerator.current_accelerator()
    } else {
        // TODO: device = torch.device("cpu")
    }
    // TODO: println!(f"Using device: {device}");
    if /* TODO: args.dataroot is None and str(args.dataset).lower() != 'fake' */ true {
        // TODO: raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % args.dataset)
    }
    let nc: usize = 3; // TODO: set based on dataset
    if ["imagenet", "folder", "lfw"].contains(&args.dataset.as_str()) {
        // folder dataset
        // TODO: dataset = dset.ImageFolder(root=args.dataroot, transform=transforms.Compose([ transforms.Resize(args.imagesize), transforms.CenterCrop(args.imagesize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])) (no Neo Theano equivalent)
        let nc = 3;
    } else if args.dataset == "lsun" {
        // TODO: classes = [ c + '_train' for c in args.classes.split(',')] (list comprehension)
        // TODO: dataset = dset.LSUN(root=args.dataroot, classes=classes, transform=transforms.Compose([ transforms.Resize(args.imagesize), transforms.CenterCrop(args.imagesize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])) (no Neo Theano equivalent)
        let nc = 3;
    } else if args.dataset == "cifar10" {
        // TODO: dataset = dset.CIFAR10(root=args.dataroot, download=True, transform=transforms.Compose([ transforms.Resize(args.imagesize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])) (no Neo Theano equivalent)
        let nc = 3;
    } else if args.dataset == "mnist" {
        // TODO: dataset = dset.MNIST(root=args.dataroot, download=True, transform=transforms.Compose([ transforms.Resize(args.imagesize), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])) (no Neo Theano equivalent)
        let nc = 1;
    } else if args.dataset == "fake" {
        // TODO: dataset = dset.FakeData(image_size=(3, args.imagesize, args.imagesize), transform=transforms.ToTensor()) (no Neo Theano equivalent)
        let nc = 3;
    }
    // TODO: assert dataset
    // TODO: dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=True, num_workers=int(args.workers)) (no Neo Theano equivalent)
    let ngpu = args.ngpu as usize;
    let nz = args.nz as usize;
    let ngf = args.ngf as usize;
    let ndf = args.ndf as usize;
    // custom weights initialization called on netG and netD
    let netG = Generator::new(nz, ngf, nc, ngpu);
    // TODO: netG.apply(weights_init)
    if args.netg != "" {
        // Load state dict for netG
    let _bytes = std::fs::read(args.netg).expect("failed to read checkpoint");
    let _state = theano_serialize::load_state_dict(&_bytes).expect("failed to load state dict");
    // TODO: apply _state to netG
    }
    // TODO: print netG
    let netD = Discriminator::new(nc, ndf, ngpu);
    // TODO: netD.apply(weights_init)
    if args.netd != "" {
        // Load state dict for netD
    let _bytes = std::fs::read(args.netd).expect("failed to read checkpoint");
    let _state = theano_serialize::load_state_dict(&_bytes).expect("failed to load state dict");
    // TODO: apply _state to netD
    }
    // TODO: print netD
    let criterion = BCELoss::new();
    let fixed_noise = Variable::new(Tensor::randn(&[args.batchsize, nz, 1, 1]));
    let real_label = 1.0_f64;
    let fake_label = 0.0_f64;
    // setup optimizer
    let mut optimizerD = Adam::new(netD.parameters(), args.lr).betas(args.beta1, 0.999);
    let mut optimizerG = Adam::new(netG.parameters(), args.lr).betas(args.beta1, 0.999);
    if args.dry_run {
        // TODO: args.niter = 1
    }
    for epoch in 0..args.niter {
        // TODO: for (i, data) in dataloader.iter().enumerate() { // TODO: implement DataLoader
        // (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        // train with real
        // netD.zero_grad()
        // TODO: let real_cpu = data[0].to(device);
        // TODO: let batch_size = real_cpu.size(0);
        // TODO: let label = Variable::new(Tensor::full(&[batch_size,], real_label));
        // TODO: let output = netD.forward(&real_cpu);
        // TODO: let errD_real = criterion.forward(&output, &label);
        // TODO: errD_real.backward();
        // TODO: let D_x = output.mean().item();
        // train with fake
        // TODO: let noise = Variable::new(Tensor::randn(&[batch_size, nz, 1, 1]));
        // TODO: let fake = netG.forward(&noise);
        // TODO: label.fill_(fake_label);
        // TODO: let output = netD.forward(&fake.detach());
        // TODO: let errD_fake = criterion.forward(&output, &label);
        // TODO: errD_fake.backward();
        // TODO: let D_G_z1 = output.mean().item();
        // TODO: let errD = errD_real + errD_fake;
        optimizerD.step();
        // (2) Update G network: maximize log(D(G(z)))
        // netG.zero_grad()
        // TODO: label.fill_(real_label);
        // TODO: let output = netD.forward(&fake);
        // TODO: let errG = criterion.forward(&output, &label);
        // TODO: errG.backward();
        // TODO: let D_G_z2 = output.mean().item();
        optimizerG.step();
        // TODO: println!("[{}/{}][{}/{}] Loss_D: {:.4} Loss_G: {:.4} D(x): {:.4} D(G(z)): {:.4} / {:.4}", epoch, args.niter, i, dataloader.len(), errD/* TODO: .item() */, errG/* TODO: .item() */, D_x, D_G_z1, D_G_z2);
        // TODO: if i % 100 == 0 {
        // TODO: vutils.save_image(real_cpu, '%s/real_samples.png' % args.outf, normalize=True) (torchvision has no Neo Theano equivalent)
        let fake = netG.forward(&fixed_noise);
        // TODO: vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (args.outf, epoch), normalize=True) (torchvision has no Neo Theano equivalent)
        if args.dry_run {
            break;
            // do checkpointing
        }
        let _bytes = theano_serialize::save_state_dict(&netG.state_dict());
    std::fs::write(format!("{}/netG_epoch_{}.pth", args.outf, epoch), _bytes).expect("failed to save checkpoint");
        let _bytes = theano_serialize::save_state_dict(&netD.state_dict());
    std::fs::write(format!("{}/netD_epoch_{}.pth", args.outf, epoch), _bytes).expect("failed to save checkpoint");
    }
}