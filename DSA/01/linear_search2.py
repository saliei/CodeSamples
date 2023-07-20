def liear_search(arr, val):
    for i in range(len(arr)):
        if arr[i] == val:
            return i
    return None

def linear_search_ordered(arr, val):
    for idx, item in enumerate(arr):
        if item == val:
            return idx
        elif item > val:
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
    high = len(arr) -1
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
    arr = [1, 4, 6, 8, 10, 12, 13, 15]
    idx1 = binary_search_recr(arr, 10, 0, len(arr)-1)
    idx2 = binary_search_iter(arr, 10)
    print(idx1)
    print(idx2)
