# numpy_pytorch.py
# Program to compare array operations between Numpy & numpy_pytorch

from __future__ import print_function
# __future__ lets your python2.7 code to inherit features of python3
import torch
import time
import numpy

# PyTorch
matrixA = torch.rand(1000,1000)
matrixB = torch.rand(1000,1000)

# current time
start_time = time.time()

# arthmetic operation
add = matrixA + matrixB
sub = matrixA - matrixB
mul = matrixA * matrixB

end_time = time.time()
print("PyTorch Execution Time: "+str(end_time - start_time)+" seconds")

# numpy
matrixA = numpy.random.rand(1000,1000)
matrixB = numpy.random.rand(1000,1000)

# current time
start_time = time.time()

# arthmetic operation
add = matrixA + matrixB
sub = matrixA - matrixB
mul = matrixA * matrixB

end_time = time.time()
print("NumpyExecution Time: "+str(end_time - start_time)+" seconds")
