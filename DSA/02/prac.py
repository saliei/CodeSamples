def bubble_sort(arr):
    ix_sorted = len(arr) - 1
    is_sorted = False
    while not is_sorted:
        is_sorted = True
        for i in range(ix_sorted):
            if arr[i] > arr[i+1]:
                arr[i], arr[i+1] = arr[i+1], arr[i]
                is_sorted = False
        ix_sorted -= 1 # NOTE
    return arr

def check_duplicate(arr):
    new_arr = [0] * (max(max(arr), len(arr))+1)
    for i in range(len(arr)):
        if new_arr[arr[i]] == 1:
            return True
        else:
            new_arr[arr[i]] = 1
    return False


if __name__ == "__main__":
    arr1 = [1, 3, 2, 8, 5, 10, 9, 8, 10, 11, 11, 11, 11]
    arr2 = [1, 2, 3, 5, 5, 8, 9, 10, 10]
    arr3 = [1, 2, 3]

    print(bubble_sort(arr1))
    print(bubble_sort(arr2))
    print(bubble_sort(arr3))

    print(check_duplicate(arr1))
    print(check_duplicate(arr2))
    print(check_duplicate(arr3))


