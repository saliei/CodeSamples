from utils.path import get_children
from utils.land import load_data
import dask.array as ds
import numpy.ma as ma
import numpy as np

# TODO: maybe work with sparse matrice using scipy
# https://docs.scipy.org/doc/scipy/reference/sparse.html
# TODO: just store non zero elements

# data = np.array([[1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                 # [0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
                 # [1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                 # [1, 0, 0, 0, 1, 1, 0, 1, 1, 1],
                 # [1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
                 # [0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
                 # [1, 0, 0, 0, 1, 1, 0, 1, 1, 1],
                 # [1, 0, 0, 0, 1, 1, 0, 1, 1, 1],
                 # [1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                 # [1, 1, 0, 0, 0, 1, 1, 1, 0, 0]], dtype=bool)

# data = np.array([[1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                 # [0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
                 # [1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                 # [1, 0, 0, 0, 1, 1, 0, 1, 1, 1],
                 # [1, 1, 0, 0, 0, 1, 1, 1, 0, 0]])

# data = np.array([[1, 0, 1],
                # [0, 1, 1]], dtype=bool)
# data = ma.masked_where(data == 0, data)
filename = "gl-latlong-1km-landcover.bsq"
data = load_data(filename)
data = data.astype(dtype=bool)
# data = ds.from_array(data)
data = data[:1000, :1000]

shape = data.shape
N, M = shape
# indices = np.indices(data.shape)
# row, col = indices

# this is equal to data
# data[row, col]

indices = np.arange(N * M)
# indices = ds.arange(N * M)
# adjacency = ds.zeros((N*M, N*M), dtype='int')
# adjacency = np.zeros((N*M, N*M), dtype='int')

# for index in indices:
    # node = np.unravel_index(index, shape)
    # # print(node)
    # children = get_children(data, node)
    # raveled_children = [np.ravel_multi_index(child, data.shape) for child in children]
    # # print(raveled_children)
    # for i in raveled_children:
        adjacency[index, i] = 1
    # print(children)


# print(adjacency)
