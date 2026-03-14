// Converted from PyTorch Python to Neo Theano Rust by torch2theano.
//
// Manual attention required:
//   - torchvision.datasets has no Neo Theano equivalent
//   - torchvision.transforms has no Neo Theano equivalent
//   - torchvision.utils has no Neo Theano equivalent
//

use theano::prelude::*;
use theano::nn::*;
use theano::optim::{Adam, SGD, Optimizer};
// use theano::data::{DataLoader, Dataset};
use theano_types::Device;
use theano_serialize::{save_state_dict, load_state_dict};
use clap::Parser;
use rand::Rng;

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
    fn new(ngpu: usize) -> Self {
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
        x.reshape(&[-1, 1]).unwrap()
    }

    fn parameters(&self) -> Vec<Variable> {
        self.main.parameters()
    }
}

fn weights_init(m: /* TODO */) {
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

    /// learning rate, default=0.0002
    #[arg(long, default_value_t = 0.0002)]
    lr: f64,

    /// beta1 for adam. default=0.5
    #[arg(long, default_value_t = 0.5)]
    beta1: f64,

    /// check a single training cycle works
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// number of GPUs to use
    #[arg(long, default_value_t = 1)]
    ngpu: usize,

    /// path to netG (to continue training)
    #[arg(long, default_value_t = String::new().into())]
    netg: String,

    /// path to netD (to continue training)
    #[arg(long, default_value_t = String::new().into())]
    netd: String,

    /// folder to output images and model checkpoints
    #[arg(long, default_value_t = ".".into())]
    outf: String,

    /// manual seed
    #[arg(long)]
    manualseed: Option<usize>,

    /// comma separated list of classes for the lsun data set
    #[arg(long, default_value_t = "bedroom".into())]
    classes: String,

    /// enables accelerator
    #[arg(long, default_value_t = false)]
    accel: bool,

}

fn main() {
    let args = Args::parse();

    println!("{}", opt);
    // TODO: try:
    // TODO: os.makedirs(opt.outf)
    // TODO: except OSError:
    if opt.manualSeed.is_none() {
        // TODO: opt.manualSeed = random.randint(1, 10000)
    }
    println!("{}", "Random Seed: ", opt.manualSeed);
    // TODO: random.seed(opt.manualSeed)
    // TODO: torch.manual_seed(opt.manualSeed)
    // TODO: cudnn.benchmark = True
    if opt.accel && torch.accelerator.is_available() {
        let device = torch.accelerator.current_accelerator(); // TODO: verify
    } else {
        let device = torch.device("cpu"); // TODO: verify
    }
    // TODO: println!(f"Using device: {device}");
    if opt.dataroot.is_none() && str(opt.dataset).lower() != "fake" {
        // TODO: raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % opt.dataset)
    }
    if ["imagenet", "folder", "lfw"].contains(&opt.dataset) {
        // folder dataset
        let dataset = dset.ImageFolder(root=opt.dataroot, transform=transforms.Compose([ transforms.Resize(opt.imageSize), transforms.CenterCrop(opt.imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])); // TODO: verify
        let nc = 3;
    } else if opt.dataset == "lsun" {
        let classes = [ c + "_train" for c in opt.classes.split(',')]; // TODO: verify
        let dataset = dset.LSUN(root=opt.dataroot, classes=classes, transform=transforms.Compose([ transforms.Resize(opt.imageSize), transforms.CenterCrop(opt.imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])); // TODO: verify
        let nc = 3;
    } else if opt.dataset == "cifar10" {
        let dataset = dset.CIFAR10(root=opt.dataroot, download=True, transform=transforms.Compose([ transforms.Resize(opt.imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])); // TODO: verify
        let nc = 3;
    } else if opt.dataset == "mnist" {
        let dataset = dset.MNIST(root=opt.dataroot, download=True, transform=transforms.Compose([ transforms.Resize(opt.imageSize), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])); // TODO: verify
        let nc = 1;
    } else if opt.dataset == "fake" {
        let dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize), transform=transforms.ToTensor()); // TODO: verify
        let nc = 3;
    }
    assert!(dataset);
    let dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers)); // TODO: verify
    let ngpu = opt.ngpu as usize;
    let nz = opt.nz as usize;
    let ngf = opt.ngf as usize;
    let ndf = opt.ndf as usize;
    // custom weights initialization called on netG and netD
    let netG = Generator::new(ngpu);
    // TODO: netG.apply(weights_init)
    if opt.netG != "" {
        // Load state dict for netG
    let _bytes = std::fs::read(opt.netG).expect("failed to read checkpoint");
    let _state = theano_serialize::load_state_dict(&_bytes).expect("failed to load state dict");
    // TODO: apply _state to netG
    }
    println!("{}", netG);
    let netD = Discriminator::new(ngpu);
    // TODO: netD.apply(weights_init)
    if opt.netD != "" {
        // Load state dict for netD
    let _bytes = std::fs::read(opt.netD).expect("failed to read checkpoint");
    let _state = theano_serialize::load_state_dict(&_bytes).expect("failed to load state dict");
    // TODO: apply _state to netD
    }
    println!("{}", netD);
    let criterion = BCELoss::new();
    let fixed_noise = Variable::new(Tensor::randn(&[opt.batchSize, nz, 1, 1]));
    let real_label = 1;
    let fake_label = 0;
    // setup optimizer
    let mut optimizerD = Adam::new(netD.parameters(), opt.lr).betas(opt.beta1, 0.999);
    let mut optimizerG = Adam::new(netG.parameters(), opt.lr).betas(opt.beta1, 0.999);
    if opt.dry_run {
        // TODO: opt.niter = 1
    }
    for epoch in 0..opt.niter {
        for (i, data) in dataloader.iter().enumerate() {
            // (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            // train with real
            netD.zero_grad();
            let real_cpu = data[0].to(device); // TODO: verify
            let batch_size = real_cpu.size(0); // TODO: verify
            let label = Variable::new(Tensor::full(&[batch_size,], real_label));
            let output = netD.forward(&real_cpu);
            let errD_real = criterion.forward(&output, &label);
            errD_real.backward();
            let D_x = output.mean().item(); // TODO: verify
            // train with fake
            let noise = Variable::new(Tensor::randn(&[batch_size, nz, 1, 1]));
            let fake = netG.forward(&noise);
            // TODO: label.fill_(fake_label);
            let output = netD.forward(&fake.detach());
            let errD_fake = criterion.forward(&output, &label);
            errD_fake.backward();
            let D_G_z1 = output.mean().item(); // TODO: verify
            let errD = errD_real + errD_fake; // TODO: verify
            optimizerD.step();
            // (2) Update G network: maximize log(D(G(z)))
            netG.zero_grad();
            // TODO: label.fill_(real_label);
            let output = netD.forward(&fake);
            let errG = criterion.forward(&output, &label);
            errG.backward();
            let D_G_z2 = output.mean().item(); // TODO: verify
            optimizerG.step();
            println!("[{}/{}][{}/{}] Loss_D: {:.4} Loss_G: {:.4} D(x): {:.4} D(G(z)): {:.4} / {:.4}", epoch, opt.niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2);
            if i % 100 == 0 {
                // TODO: vutils.save_image(real_cpu, '%s/real_samples.png' % opt.outf, normalize=True) (torchvision has no Neo Theano equivalent)
                let fake = netG.forward(&fixed_noise);
                // TODO: vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch), normalize=True) (torchvision has no Neo Theano equivalent)
            }
            if opt.dry_run {
                break;
                // do checkpointing
            }
        }
        let _bytes = theano_serialize::save_state_dict(&netG.state_dict());
    std::fs::write(format!("{}/netG_epoch_{}.pth", opt.outf, epoch), _bytes).expect("failed to save checkpoint");
        let _bytes = theano_serialize::save_state_dict(&netD.state_dict());
    std::fs::write(format!("{}/netD_epoch_{}.pth", opt.outf, epoch), _bytes).expect("failed to save checkpoint");
    }
}
