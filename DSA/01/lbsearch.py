def linear_search(arr, val):
    for idx, itm in enumerate(arr):
        if itm == val:
            return idx
    return None

def linear_search_ordered(arr, val):
    for idx, itm in enumerate(arr):
        if itm == val:
            return idx
        elif itm > val:
            break
    return None

def binary_search_recr(arr, val, low, high):
    if high >= low:
        mid = (low + high) // 2
        if arr[mid] == val:
            return mid
        elif arr[mid] < val:
            return binary_search_recr(arr, val, mid+1, high)
        else:
            return binary_search_recr(arr, val, low, mid-1)
    else:
        return None

def binary_search_iter(arr, val):
    low = 0
    high = len(arr) - 1
    mid = 0
    while hgih >= low:
        mid = (low + high) // 2
        if arr[mid] == val:
            return mid
        elif arr[mid] > val:
            high = mid-1
        else:
            low = mid+1
    return None

if __name__ == "__main__":
    arr = [1, 3, 5, 6, 8, 10]
    idx = linear_search_ordered(arr, 10)
    print(idx)
