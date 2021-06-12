---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import numpy as np
import torch
import torch.nn as nn
```

```python
class SimpleCouplingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = torch.nn.Sequential(
            torch.nn.Linear(1, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
            torch.nn.Tanh()
        )
    def forward(self, x):
        return self.s(x)
```

```python
x1 = torch.tensor(np.random.random(10), dtype=torch.float32)
x2 = torch.tensor(np.random.random(10), dtype=torch.float32)
torch.stack((x1,x2), dim=-1).shape
```

```python
s = SimpleCouplingLayer()
```

```python
s(x2.unsqueeze(-1)).squeeze()
```
