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
import torch
from matplotlib import pyplot as plt
import numpy as np
```

```python
def mypdf(x):
    return 1/np.sqrt(2*np.pi) * torch.exp(-0.5*(x**2))
```

```python
dist = torch.distributions.normal.Normal(
    torch.from_numpy(np.zeros((3,4,5))), torch.from_numpy(np.ones((3,4,5)))
)
```

```python
torch.abs
```

```python
s = torch.rand((3,4,5))
assert torch.all(torch.abs(dist.log_prob(s)-torch.log(mypdf(s))) < 1e-6)
```

```python
s = np.random.random()
assert torch.all(dist.log_prob(torch.tensor([s])) - torch.log(mypdf(torch.tensor([s]))) < 1e-7)
```

```python
class SimpleNormal():
    def __init__(self, loc, var):
        self.dist = torch.distributions.normal.Normal(
            torch.flatten(loc), torch.flatten(var)
        )
        self.shape = loc.shape
    def log_prob(self, x):
        # reshape tensor from　(B, s1,s2,s3,...)  to (B, s1 * s2 * s3 * ...)
        x_reshape = x.reshape(x.shape[0], -1)
        logp = self.dist.log_prob(x_reshape)
        assert x_reshape.shape == logp.shape
        return torch.sum(logp, dim=1)
    def sample_n(self, batch_size):
        x = self.dist.sample((batch_size, ))
        return x.reshape(batch_size, *self.shape)
```

```python
normal_prior = SimpleNormal(
    torch.zeros((3,4,5)),
    torch.ones((3,4,5)),
)

z = normal_prior.sample_n(17)
print(z.shape)
print(normal_prior.log_prob(z).shape)
```

```python

```
