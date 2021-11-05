#!/usr/bin/env python

from collections import deque
from utils.land import load_data
import numpy as np
import dask.array as ds
from utils.land import load_data, get_index
#from dask.distributed import Client
#from dask.diagnostics import ProgressBar

# from utils.land import ...
# from utils.path import dfs, ...

# unit tests

# TODO: make path unique

# https://www.interviewbit.com/tutorial/depth-first-search/
# Since we are using a list as opposed to a set in Python to keep track of visited vertices, the search to see if a vertex has already been visited has a linear runtime as opposed to constant runtime
# print all the nodes for the path in a constant time, should consider the size of the path
# store raveled indices instead of a tuple maybe in a set for large grids
# neighbors instead of children
# https://www.baeldung.com/cs/iterative-deepening-vs-depth-first-search
# test ravel and unravel for speed
# test stack limit, ulimit -s unlimited
# distributed and parallel version with dask, client, ... to benchmark...

# TODO: see stack and deque and queue efficiencies?
# TODO: validity and goal checking in a separate function
# DFS, IDDLS, BFS, Dijkstra?
# BFS in a graph with no weight and direction is the same as Dijkstra(weight=1, one direction)
# DFS doesn't find the shortest path necessarily!! simple test: start=(0, 6), end=(2, 5)
# https://stackoverflow.com/questions/14784753/shortest-path-dfs-bfs-or-both
# TODO: guess the max_depth depending on the distance between points to optimize it a bit.
# strange cases: start=(0, 0), end=(0, 2)
# make checking for validity simple, by using hard cases for edges , etc.
# https://eddmann.com/posts/using-iterative-deepening-depth-first-search-in-python/
# https://www.baeldung.com/cs/iterative-deepening-vs-depth-first-search

# TODO: find a way to work with large grids!
# TODO: maybe at end we have to coarsen the map!
# if you can't find a path between two land points you have to fly!

# algorithm name
# ITERATIVE DEEPENING DEPTH LIMITED DEPTH FIRST SEARCH

# in a satellite navigation system this will cost us, and it's not
# a very smart way of searching paths


#originmap[x+1,y]=np.ravel_multi_index([x,y], (max_val,max_val))

# client = Client(n_workers=4, processes=False, threads_per_worker=1)


# for this case np is much faster than ds
# arr = ds.from_array(np_arr)


# mapfile = "gl-latlong-1km-landcover.bsq"
# mapdata = load_data(mapfile, lazyload=False)
# mapdata = mapdata[:1000, :2000]
# mapdata[mapdata > 0] = 1



def is_target(node, target):
    if node == target:
        return True

    return False


def is_valid(grid, node):
    i, j = node
    if i < 0 or j < 0:
        return False
    if i >= grid.shape[0] or j >= grid.shape[1]:
        return False
    if grid[node] == 0:
        return False

    return True


# boundary condition?
def get_children(grid, node):
    i, j = node
    children = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
    children = [child for child in children if is_valid(grid, child)]

    return children


# this is iterative but not recursive
# visited should be a set or something
def dfs_iterative(grid, source, target):
    explore = deque([source])
    visited = deque([source])

    while explore:
        node = explore.pop()
        if is_target(node, target):
            return visited, True
        children = get_children(grid, node)
        for child in children:
            if child not in visited:
                explore.append(child)
                visited.append(child)

    return visited, False


def dfs_recursive(grid, node, target, visited=deque()):
    visited.append(node)
    if is_target(node, target):
        return visited, True
    children = get_children(grid, node)
    for child in children:
        if child not in visited:
            visited, result = dfs_recursive(grid, child, target, visited)
            # return only exits current function not all others in the stack!
            if result:
                return visited, True

    return visited, False


# visited nodes are more frequent
# iterative depth limited depth-first search?
def dldfs(grid, source, target, limit=100):
    explore = deque([source])
    visited = deque([source])

    depth = 0
    if depth >= limit:
        print("Depth limit reached.")
        return visited, "CUTOFF"

    while explore:
        node = explore.pop()
        if is_target(node, target):
            return visited, True
        children = get_children(grid, node)
        for child in children:
            if child not in visited:
                explore.append(child)
                visited.append(child)
        depth += 1

    return visited, False


def iddldfs(grid, source, target):
    max_depth = 1000
    for limit in range(max_depth):
        visited, result = dldfs(grid, source, target, limit)
        if result != "CUTOFF":
            return visited, result

