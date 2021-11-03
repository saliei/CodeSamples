#!/usr/bin/env python

import numpy as np

# from utils.land import ...
# from utils.path import ...

# unit tests

# TODO: make path unique



arr = np.array([[1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                [0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
                [1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 1, 1, 0, 1, 1, 1],
                [1, 1, 0, 0, 0, 1, 1, 1, 0, 0]])


source = (0, 0)
target = (1, 1)

start = source
end = target
goal = target

maxDepth = 40


path = []

def is_target(node, target):
    if node == target:
        return True
    return False

def is_valid(grid, node):
    i, j = node
    # if (i < 0 or j < 0) or \
            # (i >= grid.shape[0] or j >= grid.shape[1]) or \
            # grid[node] == 0:
        # return False
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



def dls(grid, road, target, depth, visited):
    node = road[-1]
    if is_target(node, target):
        return road
    if depth <= 0:
        return None

    children = get_children(grid, node)
    for child in children:
        if child not in road:
            visited.append(child)
            new_path = path.copy()
            new_path.append(child)
            next_path = dls(grid, new_path, target, depth-1, visited)
            if next_path is not None:
                return next_path


def ids(grid, source, target):
    max_depth = 10
    visited_set = []
    visited = []
    for depth in range(max_depth):
        visited.append(source)
        road = dls(grid, [start], target, depth, visited)
        if visited not in visited_set:
            visited_set.append(visited.copy())
        if road is None:
            continue
        return visited_set, road
    return visited_set, None


if __name__ == "__main__":
    # path, result = find_path(arr, source, target)
    # vis = np.full(arr.shape, -1) 
    # result = __has_path(arr, vis)
        
    _set, result = ids(arr, source, target)
    print(_set)
    print(result)

    # print(path)
    # print(result)
