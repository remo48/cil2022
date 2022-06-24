import numpy as np

arr = np.array( [[ 1, 2, 3],
                 [ 4, 2, 5]] )
 
print("Array is of type: ", type(arr))
 
# Printing array dimensions (axes)
print("No. of dimensions: ", arr.ndim)
 
# Printing shape of array
print("Shape of array: ", arr.shape)
 
# Printing size (total number of elements) of array
print("Size of array: ", arr.size)
 
# Printing type of elements in array
print("Array stores elements of type: ", arr.dtype)

import torch
print(torch.cuda.is_available())
torch.cuda.device_count()