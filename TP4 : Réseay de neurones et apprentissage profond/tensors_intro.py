from __future__ import print_function
import torch
import numpy as np

########################################################################
# Doc : https://pytorch.org/docs/stable/torch.html#tensors
########################################################################


### Declarations

# Declarer (sans initialiser) un Tensor pyTorch (3x2)
x = torch.empty(3, 2, dtype=torch.int32)
print(x)

# Declarer et initialiser un Tensor pyTorch (3x2)
x =  torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
print(x)

# Declarer un Tensor (4,2) dont tous les elements sont initialise a zero
x = torch.zeros(4, 2, dtype=torch.int32)
print(x)

# Declarer un Tensor (5x3) aleatoirement initialise
x = torch.rand(5, 3)
print(x)

# Declarer un Tensor (3x3) dont les valeurs sont initialisees entre 0 et 1 suivant une loi normale
tensor_normal = torch.randn(3, 3)
tensor_normalized = (tensor_normal - tensor_normal.min()) / (tensor_normal.max() - tensor_normal.min())
print(x)

# Declarer un batch de 10 Tensor (3x3) initilise aleatoirement
batch = torch.rand(10, 3, 3)
print(batch)

### Operations

# Additionner un Tensor (2x2) avec un scalaire
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
z = torch.add(x, 1)
print(z)

# Additionner deux Tensors (2x2)
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
y = torch.tensor([[5, 6], [7, 8]], dtype=torch.int32)
z = torch.add(x, y)
print(x)
print(y)
print(z)

# Mutliplier un Tensor (5x2) par un Tensor (2x3)
a = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=torch.int32)
b = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
c = torch.mm(a, b)
print(a)
print(b)
print(c)

# Mutliplier 2 batch de 3 Tensor (2x2)
batch1 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=torch.int32)
batch2 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=torch.int32)
r = torch.bmm(batch1, batch2)
print(batch1)
print(batch2)
print(r)
print(batch1[0])
print(batch2[0])
print(r[0])



### Slincing Indexing Joining Mutating

# Declarer un Tensor de 9 elements puis le formater en Tensor (3x3)
v = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int32)
v = torch.reshape(v, (3, 3))
print(v)

# Concatener deux Tensor (2x2)
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.int32)
c = torch.cat((a, b), 0)
print(c)

# Declarer un Tensor (10,5) et afficher sa premiere colonne
a = torch.rand(10, 5)
print(a)
print(a[:, 0])



