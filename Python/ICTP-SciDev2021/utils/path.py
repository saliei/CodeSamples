"""Module file for exercise 1 for the ICTP SciDev2021 workshop.
For the problem description refere to:
https://corbetta.phys.tue.nl/pages/ictp-l21-121.html

TODO:
    * Fix long search time for large grids.
        * Smart search, only search a square grid containing the target and source nodes.
        * Test dask arrays for parallel and distributed search.
        * Coarsen the grid.
    * Expand to find a path if the source and target are in water.
    * Use a weighted graph, depending on the landscape with an appropriate weight.

"""
from utils.land import load_data, get_index
from typing import Union, Tuple, Deque, List
from collections import deque
import numpy as np


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


def get_children(grid: Union[np.ndarray, np.memmap], node: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Function to get the neighbor nodes of the current node.

    This function will explore the surrounding nodes, and will get the valid neighbors.

    TODO:
        * The boundary condition should be a spherical mapping.

    Args:
        grid(Union[np.ndarray, np.memmap]): The matrix for the map.
        node (Tuple[int, int]): The tuple for the index of the node.

    Returns:
        List[Tuple[int, int]]: A list containing the tuples of indices for the neighbor nodes.

    """
    i, j = node
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

