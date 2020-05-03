# hello_tensor.py
# First baby step towards Deep Learning using PyTorch

from __future__ import print_function
# __future__ lets your python2.7 code to inherit features of python3
import torch

print ("> hello_tensor \n")

# uninitialized matrix 
print ("uninitialized matrix")
matrix = torch.empty(2,2)
print(matrix)

# randomly initialized matrix 
print ("randomly initialized matrix")
matrix = torch.rand(2,2)
print(matrix)

# initialize matrix with zeros and dtype as long
print ("initialize matrix with zeros and dtype as long")
matrix = torch.zeros(2,2,dtype=torch.long)
print(matrix)

# contruct a tensor directory from data
print ("contruct a tensor directory from data")
matrix = torch.tensor([7, 7, 7])
print(matrix)

# create a tensor based on an existing tensor
matrix = matrix.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(matrix)

matrix = torch.randn_like(matrix, dtype=torch.float)    # override dtype!
print(matrix)                                      # result has the same size