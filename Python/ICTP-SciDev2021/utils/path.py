#!/usr/bin/env python
"""Module file for exercise 1 for the ICTP SciDev2021 workshop.
For the problem description refere to:
https://corbetta.phys.tue.nl/pages/ictp-l21-121.html

TODO:
    * Fix long search time for large grids.

"""

from utils.land import load_data, get_index
from typing import Union, Tuple, Deque, List
from collections import deque
import dask.array as ds
import numpy as np
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
# TODO: make a graph out of the grid to prun a lot of unneeded nodes with probably adjancy matrix!
# TODO: a class that makes a graph out of a map, if weighted takes the different landscape type into account!
# TODO: if point is on water, children should be only grid points where it is 0, so go by boat, walk, fly, swim

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



def is_target(node: Tuple[int, int], target: Tuple[int, int]) -> bool:
    """Check if the current node is the target node.

    Args:
        node (Tuple[int, int]): A tuple for the index of the current node.
        target (Target[int, int]): A tuple for the index of the target node.

    Returns:
        bool: True if the node is the target, False otherwise.

    """
    if node == target:
        return True

    return False


def is_valid(grid: Union[np.ndarray, np.memmap] , node: Tuple[int, int]) -> bool:
    """Check if the node exploring is a valid candidate.

    If the node fallse outside of the grid boundary or the value of 
    the grid for the node is 0 it is not a valid candidate.

    Args:
        grid (Union[np.ndarray, np.memmap]): The matrix for the map.
        node (Tuple[int, int]): The tuple for the index of the node.

    Returns:
        bool: True if the node is valid candidate, False otherwise.

    """
    i, j = node
    if i < 0 or j < 0:
        return False
    if i >= grid.shape[0] or j >= grid.shape[1]:
        return False
    if grid[node] == 0:
        return False

    return True


# boundary condition?
def get_children(grid: Union[np.ndarray, np.memmap], node: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Function to get the neighbor nodes of the current node.

    This function will explore the surrounding nodes, and will get the valid neighbors.

    Args:
        grid(Union[np.ndarray, np.memmap]): The matrix for the map.
        node (Tuple[int, int]): The tuple for the index of the node.

    Returns:
        List[Tuple[int, int]]: A list containing the tuples of indices for the neighbor nodes.

    """
    i, j = node
    # if it's water can't walk
    # no direction from 0 to 1
    if grid[node] == 0:
        children = []
    else:
        children = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
        children = [child for child in children if is_valid(grid, child)]
    return children


def dfs_iterative(grid: Union[np.ndarray, np.memmap], source: Tuple[int, int], target: Tuple[int, int]) -> Union[np.ndarray, bool]:
    """Iterative Depth-First Search algorithm.

    This is the iterative DFS algorithm, it will explore nodes and find a path 
    from source to target, if it exists, iteratively.

    TODO:
        * Make visited nodes a unique collection, e.g. Set.

    Args:
        grid (Union[np.ndarray, np.memmap]): The matrix for the map.
        source (Tuple[int, int]): The starting point for the path.
        target (Tuple[int, int]): The ending point for the path.

    Returns:
        Union[np.ndarray, bool]: The first element of the pack is the visited nodes, 
            the second is a boolean, True if path exists, False if it doesn't.

    """
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


def dfs_recursive(grid: np.ndarray, node: Tuple[int, int], target: Tuple[int, int], visited:Deque[Tuple[int, int]] = deque()) -> Union[np.ndarray, bool]:
    """Recursive Depth-First Search algorithm.

    This is the recursive DFS algorithm, it will explore nodes and find a path 
    from source to target, if it exists, recursively.

    TODO:
        * Make visited nodes a unique collection, e.g. Set.

    Args:
        grid (Union[np.ndarray, np.memmap]): The matrix for the map.
        source (Tuple[int, int]): The starting point for the path.
        target (Tuple[int, int]): The ending point for the path.
        visited (Deque[Tuple[int, int]]): A deque of tuple containing the indices of visited nodes.

    Returns:
        Union[np.ndarray, bool]: The first element of the pack is the visited nodes, 
            the second is a boolean, True if path exists, False if it doesn't.

    """
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


def dldfs(grid: np.ndarray, source: Tuple[int, int], target: Tuple[int, int], limit: int = 100) -> Union[np.ndarray, bool]:
    """Depth-Limited Depth-First Search algorithm.

    This is the depth-limited DFS algorithm, it will explore nodes and find a path 
    from source to target, if it exists. Contrary to DFS, depth-limited DFS will first search 
    all branches with a specific depth first, if target isn't reached, it will search other branches
    with different depth. Compared to DFS, DLDFS has more visited nodes.


    Args:
        grid (Union[np.ndarray, np.memmap]): The matrix for the map.
        source (Tuple[int, int]): The starting point for the path.
        target (Tuple[int, int]): The ending point for the path.
        limit (int, optional): Depth limit for branch exploration. Defaults to 100.

    Returns:
        Union[np.ndarray, bool]: The first element of the pack is the visited nodes, 
            the second is a boolean, True if path exists, False if it doesn't.

    """
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


def iddldfs(grid: np.ndarray, source: Tuple[int, int], target: Tuple[int, int]) -> Union[np.ndarray, bool]:
    """Iterative Deepening Depth-Limited Depth-First Search algorithm.

    This is the iterative deepening depth-limited DFS algorithm, it will explore nodes and find the shortest path 
    from source to target, if it exists. Compared to DLDFS algorithm, the deepening DLDFS will start searching all 
    branches from 1 to a maximum depth, this will ensure that the found path will be the shortest one.

    Args:
        grid (Union[np.ndarray, np.memmap]): The matrix for the map.
        source (Tuple[int, int]): The starting point for the path.
        target (Tuple[int, int]): The ending point for the path.

    Returns:
        Union[np.ndarray, bool]: The first element of the pack is the visited nodes, 
            the second is a boolean, True if path exists, False if it doesn't.

    """
    max_depth = 1000
    for limit in range(max_depth):
        visited, result = dldfs(grid, source, target, limit)
        if result != "CUTOFF":
            return visited, result

