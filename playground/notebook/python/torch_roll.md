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
import numpy as np
import torch

if torch.cuda.is_available():
    torch_device = 'cuda'
    float_dtype = np.float32 # single
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch_device = 'cpu'
    float_dtype = np.float64 # double
    torch.set_default_tensor_type(torch.DoubleTensor)
print(f"TORCH DEVICE: {torch_device}")

lattice_shape = (3, 3)
```

```python
phi_ex1 = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7,8,9],
])

phi_ex2 = np.array([
    [10, 11, 12],
    [13,14,15],
    [16,17,18],
])

cfgs = torch.from_numpy(np.stack((phi_ex1, phi_ex2), axis=0)).to(torch_device)
```

```python
assert cfgs.shape == (2,*lattice_shape)
```

```python
torch.roll(cfgs, -1, 1)
```

```python
torch.roll(cfgs, 1, 1)
```
