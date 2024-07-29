from __future__ import print_function
import torch
import numpy as np

# Declarer et initialiser un Tensor x de taille 2 × 2
x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
print(x)

# D ́eclarer et initialiser un Tensor tel que y = x + 5
y = torch.add(x, 5)
print(y)

# D ́eclarer et initialiser un Tensor tel que z = (3 × y2)/x
z = torch.div(torch.mul(torch.pow(y, 2), 3), x)
print (z)

# D ́eclarer et initialiser un Tensor tel que o = mean(z)
o = torch.mean(z)
print (o)

# Calculer la derivee ∂o/∂x a la main 


# calculer la derivee ∂o/∂x a l'aide d'autograd
o.backward()
print(x.grad)
