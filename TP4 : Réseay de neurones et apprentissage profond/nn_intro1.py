import torch

# Cr ́eer un block de convolution 1D ayant en entr ́ee 1 canal et 2 en sortie (afficher sesparam`etres),
conv1d = torch.nn.Conv1d(1, 2, kernel_size=1)
print(conv1d.weight)
print(conv1d.bias)
print('--------------------------')


# D ́eclarer x, un Tensor 1D d’un seul  ́el ́ement al ́eatoirement initialis ́e,
x = torch.tensor([[[1.]]], requires_grad=True)
print(x)
print('--------------------------')


# Alimenter votre couche de convolution avec votre Tensor x et conserver la sortie obtenue dans un Tensor y
y = conv1d(x)
print(y)
print('--------------------------')


# D ́eclarer o, un Tensor 1D initialis ́e comme  ́etant la somme des  ́el ́ements de y
o = torch.sum(y)

# V ́erifier votre calcul de ∂o `a l’aide d’autograd
o.backward()
print(x.grad)