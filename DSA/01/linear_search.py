# linear/binary search on ordered array

def linear_search(arr, val):
    for idx, item in enumerate(arr):
        if item == val:
            return idx
        elif item > val:
            break
    return None
    

def binary_search_recursive(arr, val, idx_start, idx_end):
    # the return criteria is if the val in the middle! the array is ordered!

    if idx_end >= idx_start:
        mid = (idx_start + idx_end) // 2
        if arr[mid] == val:
            return mid
        elif arr[mid] < val:
            return binary_search_recursive(arr, val, mid+1, idx_end)
        else:
            return binary_search_recursive(arr, val, idx_start, mid-1)
    else:
        return None

def binary_search_iter(arr, val):
    low = 0
    high = len(arr) - 1
    mid = 0
    while high >= low:
        mid = (low + high) // 2
        if arr[mid] == val:
            return mid
        elif arr[mid] > val:
            high = mid - 1
        else:
            low = mid + 1
    return None


if __name__ == "__main__":
    arr = [1, 2, 4, 6, 7, 10]

    idx1 = linear_search(arr, 7)
    idx2 = linear_search(arr, 8)

    idx3 = binary_search_recursive(arr, 4, 0, len(arr)-1)
    idx4 = binary_search_recursive(arr, 5, 0, len(arr)-1)

    idx5 = binary_search_iter(arr, 2)
    idx6 = binary_search_iter(arr, 7)
    idx7 = binary_search_iter(arr, 5)

    print(idx1)
    print(idx2)

    print(idx3)
    print(idx4)

    print(idx5)
    print(idx6)
    print(idx7)
