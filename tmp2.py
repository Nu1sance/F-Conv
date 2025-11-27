import torch

a = torch.rand(2, 2, 4)
b = torch.rand(3, 3, 4)

print(f'a is {a}')
print(f'b is {b}')
c = torch.einsum('ijk, mnk -> mnij', a, b) # noqa
print(f'c shape: {c.shape}')
print(f'c is {c}')
