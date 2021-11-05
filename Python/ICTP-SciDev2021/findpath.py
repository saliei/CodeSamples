#!/usr/bin/env python

import numpy as np
# for zero like arrays
import dask.array as ds
from collections import deque
from utils import load_data
# from resource import setrlimit


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

arr = np.array([[1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                [0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
                [1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 1, 1, 0, 1, 1, 1],
                [1, 1, 0, 0, 0, 1, 1, 1, 0, 0]])

# mapfile = "gl-latlong-1km-landcover.bsq"
# mapdata = load_data(mapfile, lazyload=False)
# mapdata = mapdata[:1000, :2000]
# mapdata[mapdata > 0] = 1

# source = (1, 7)
# target = (2, 5)
source = (0, 1)
target = (2, 0)
# target = (3, 3)
# source = (10, 100)
# target = (400, 1500)


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


def get_children(grid, node):
    i, j = node
    children = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
    children = [child for child in children if is_valid(grid, child)]

    return children


def has_path(grid, visited, node, target):
    if visited[node] >= 0:
        return False
    visited[node] = 0

    children = get_children(grid, node)
    for child in children:
        if is_target(node, target) or has_path(grid, visited, child, target):
            visited[node] = 1
            path.append(node)

    return path, visited[node] == 1


def find_path(grid, source, target):
    visited = np.full(grid.shape, -1)
    path, result = has_path(grid, visited, source, target)

    return path, result


# def dls(grid, road, target, depth, visited):
    # node = road[-1]
    # if is_target(node, target):
        # return road
    # if depth <= 0:
        # return None

    # children = get_children(grid, node)
    # for child in children:
        # if child not in road:
            # visited.append(child)
            # new_path = path.copy()
            # new_path.append(child)
            # next_path = dls(grid, new_path, target, depth-1, visited)
            # if next_path is not None:
                # return next_path


# def ids(grid, source, target):
    # max_depth = 10
    # visited_set = []
    # visited = []
    # for depth in range(max_depth):
        # visited.append(source)
        # road = dls(grid, [start], target, depth, visited)
        # if visited not in visited_set:
            # visited_set.append(visited.copy())
        # if road is None:
            # continue
        # return visited_set, road
    # return visited_set, None


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
        print("target reached")
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
        # return visited, False
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


if __name__ == "__main__":
    # path, result = find_path(arr, source, target)
    # vis = np.full(arr.shape, -1) 
    # result = __has_path(arr, vis)
        
    # _set, result = ids(arr, source, target)
    # print(_set)
    # print(result)

    # print(path)
    # print(result)

    # visited, result = dfs2(arr, source, target)
    # visited, result = dfs2(mapdata, source, target)
    # print(result)
    # print(visited)
    
    # vis = np.full(arr.shape, False)
    # visited, result = rec_dfs2(arr, vis, source, target)
    # result = rec_dfs2(arr, vis, source, target)
    # print(result)
    # print(visited)

    # visited, result = rec_dfs4(arr, source, target)
    # print(result)
    # print(visited)

    # result = dls2(arr, source, target)
    # print(result)

    # visited, result = dls5(arr, source, target)
    # print(result)
    # print(visited)

    visited, result = iddldfs(arr, source, target)
    print(result)
    print(visited)
