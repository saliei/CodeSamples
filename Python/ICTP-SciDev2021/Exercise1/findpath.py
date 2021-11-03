#!/usr/bin/env python

import numpy as np

# from utils.land import ...
# from utils.path import ...

# unit tests



arr = np.array([[1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                [0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
                [1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 1, 1, 0, 1, 1, 1],
                [1, 1, 0, 0, 0, 1, 1, 1, 0, 0]])


source = (0, 0)
target = (0, 1)

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
    if (i < 0 or j < 0) or \
            (i >= grid.shape[0] or j >= grid.shape[1]) or \
            grid[node] == 0:
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
    return visited, visited[node] == 1

def find_path(grid, source, target):
    visited = np.full(grid.shape, -1)
    visited, has = has_path(grid, visited, source, target)
    return visited, has


def DLS(graph, path, goal, depth, expanded):
    node = path[-1]

    if node == goal:
        return path

    if depth <= 0:
        return None

    for child in range(len(graph[node])):
        if graph[node][child] != 0:
            if child not in path:
                expanded.append(child)
                new_path = path.copy()
                new_path.append(child)
                next_path = DLS(graph, new_path, goal, depth - 1, expanded)
                if next_path is not None:
                    return next_path


def IDS_Algorithm(graph, start, goal):
    expandedSet = []
    expanded = []
    for i in range(maxDepth):
        expanded.clear()
        expanded.append(start)
        path = DLS(graph, [start], goal, i, expanded)
        if expanded not in expandedSet:
            expandedSet.append(expanded.copy())
        if path is None:
            continue
        return expandedSet, path
    return expandedSet, None


if __name__ == "__main__":
    visited, result = find_path(arr, source, target)
    # vis = np.full(arr.shape, -1) 
    # result = __has_path(arr, vis)

    print(visited)
    print(result)
