import numpy as np
import torch


class ScalarPhi4Action:
    def __init__(self, M2, lam):
        self.M2 = M2
        self.lam = lam

    def __call__(self, cfgs):
        """
        cfgs.shape == (batch_size, L, L)
        """
        action_density = self.M2 * cfgs ** 2 + self.lam * cfgs ** 4
        dims = range(1, cfgs.ndim)
        for mu in dims:
            action_density += 2 * cfgs ** 2
            action_density -= cfgs * torch.roll(cfgs, -1, mu)
            action_density -= cfgs * torch.roll(cfgs, 1, mu)
        return torch.sum(action_density, dim=tuple(dims))


def create_cfgs(L):
    rng = np.random.default_rng(2021)
    lattice_shape = (L, L)
    phi_ex1 = rng.normal(size=(lattice_shape)).astype(np.float32)
    phi_ex2 = rng.normal(size=(lattice_shape)).astype(np.float32)
    cfgs = np.stack((phi_ex1, phi_ex2), axis=0)
    return cfgs


L = 8
cfgs = create_cfgs(L)
pyaction1 = ScalarPhi4Action(M2=1.0, lam=1.0)
out1 = pyaction1(torch.from_numpy(cfgs)).detach().numpy()
pyaction2 = ScalarPhi4Action(M2=-4.0, lam=8)
out2 = pyaction2(torch.from_numpy(cfgs)).detach().numpy()
