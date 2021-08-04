---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
from matplotlib import pyplot as plt
import torch
import numpy as np
```

複雑な形状を持つ確率分布のサンプルを生成するために比較的単純な確率分布からサンプルを生成し，変数変換で所望の分布を得る Normalizing Flows  
を効率よくするための手法を Lattice Field Theory に適用する.

```python
batch_size= 2 ** 14
u1 = np.random.random(size=batch_size)
u2 = np.random.random(size=batch_size)
z1 = np.sqrt(-2*np.log(u1)) * np.cos(2*np.pi*u2)
z2 = np.sqrt(-2*np.log(u1)) * np.sin(2*np.pi*u2)
fig, ax = plt.subplots()
ax.hist2d(
    z1,
    z2,
    bins=30,
    range=[
        [-3., 3.],
        [-3., 3.],
    ]
)
ax.set_aspect("equal")
```

```python
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
        x = self.distribution.sample((batch_size, ))
        return x
```

```python
normal_prior = SimpleNormal(torch.zeros((3,4,5)), torch.ones((3,4,5)))
z = normal_prior.sample_n(17)
print(z.shape)
normal_prior.log_prob(z)
```
