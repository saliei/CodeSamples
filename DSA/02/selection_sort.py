def selection_sort(arr):
    i = 0
    while i < len(arr) - 1:
        min_idx = i
        for j in range(len(arr)-i):
            if arr[i+j] < arr[i]:
                min_idx = i+j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
        i += 1
    return arr

if __name__ == "__main__":
    arr1 = [1, 4, 3, 2, 5, 5]
    arr2 = [1, 2, 2, 3, 5, 6]

    print(selection_sort(arr1))
    print(selection_sort(arr2))
