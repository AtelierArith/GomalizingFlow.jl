import itertools

import numpy as np
import torch

L = 8
lattice_shape = (L, L)
M2 = -4.0
lam = 8.0

n_layers = 16
hidden_sizes = [8, 8]
kernel_size = 3
inC = 1
outC = 2
use_final_tanh = True

batch_size = 64


def make_checker_mask(shape, parity):
    checker = torch.ones(shape, dtype=torch.uint8) - parity
    checker[::2, ::2] = parity
    checker[1::2, 1::2] = parity
    return checker


assert torch.all(
    make_checker_mask(lattice_shape, 0)
    == torch.from_numpy(
        np.array(
            [
                [0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 0],
            ]
        )
    )
)


class SimpleNormal:
    def __init__(self, loc, var):
        self.distribution = torch.distributions.normal.Normal(
            loc,
            var,
        )
        self.shape = loc.shape

    def log_prob(self, x):
        logp = self.distribution.log_prob(x)
        return torch.sum(logp, dim=tuple(range(1, logp.ndim)))

    def sample_n(self, batch_size):
        x = self.distribution.sample((batch_size,))
        return x


torch.manual_seed(12345)


class MyAffineCoupling(torch.nn.Module):
    def __init__(self, net, *, mask_shape, parity):
        self.parity = parity
        super(MyAffineCoupling, self).__init__()
        self.mask = make_checker_mask(mask_shape, parity)
        self.mask_flipped = 1 - self.mask

        self.net = net

    def forward(self, x):  # (B, C, H, W)
        x_frozen = self.mask * x  # \phi_2
        x_active = self.mask_flipped * x  # \phi_1
        net_out = self.net(x_frozen.unsqueeze(1))
        s, t = net_out[:, 0], net_out[:, 1]
        # ((exp(s(phi_)))\phi_1 + t(\phi_2), \phi_2) を一つのデータとして
        fx = self.mask_flipped * t + x_active * torch.exp(s) + x_frozen
        axes = range(1, len(s.size()))
        logJ = torch.sum(self.mask_flipped * s, dim=tuple(axes))
        return fx, logJ

    def reverse(self, fx):
        fx_frozen = self.mask * fx  # phi_2'
        fx_active = self.mask_flipped * fx  # phi_1'
        net_out = self.net(fx_frozen.unsqueeze(1))
        s, t = net_out[:, 0], net_out[:, 1]
        x = (fx_active - self.mask_flipped * t) * torch.exp(-s) + fx_frozen
        axes = range(1, len(s.size()))
        logJ = torch.sum(self.mask_flipped * (-s), dim=tuple(axes))
        return x, logJ


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


module_list = []
for i in range(n_layers):
    parity = i % 2
    sizes = [inC, *hidden_sizes, outC]
    padding = kernel_size // 2
    net = []
    for s, s_next in pairwise(sizes):
        net.append(
            torch.nn.Conv2d(
                s, s_next, kernel_size, padding=padding, padding_mode="circular"
            )
        )
        net.append(torch.nn.LeakyReLU())
    if use_final_tanh:
        net[-1] = torch.nn.Tanh()
    net = torch.nn.Sequential(*net)
    coupling = MyAffineCoupling(net, mask_shape=lattice_shape, parity=parity)
    module_list.append(coupling)

prior = SimpleNormal(torch.zeros(lattice_shape), torch.ones(lattice_shape))
my_layers = torch.nn.ModuleList(module_list)

my_model = {"layers": my_layers, "prior": prior}


def apply_affine_flow_to_prior(r, aff_coupling_layers, *, batch_size):
    z = r.sample_n(batch_size)
    logq = r.log_prob(z)
    x = z
    for lay in aff_coupling_layers:
        x, logJ = lay.forward(x)
        logq = logq - logJ
    return x, logq  # 点 x における `\log(q(x))` の値を計算


apply_affine_flow_to_prior(prior, my_layers, batch_size=batch_size)


def applyflow(z):
    x = z
    for lay in my_layers:
        x, _ = lay.forward(x)
    return x
