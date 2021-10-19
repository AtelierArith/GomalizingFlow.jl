import numpy as np


def create_cfgs(L):
    rng = np.random.default_rng(2021)
    lattice_shape = (L, L)
    phi_ex1 = rng.normal(size=(lattice_shape)).astype(np.float32)
    phi_ex2 = rng.normal(size=(lattice_shape)).astype(np.float32)
    cfgs = np.stack((phi_ex1, phi_ex2), axis=0)
    return cfgs
