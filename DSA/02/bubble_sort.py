def swap1(val1, val2):
    tmp = val1
    val1 = val2
    val2 = tmp
    return val1, val2

def swap2(val1, val2):
    return val2, val1

# in each pass the highest unsorted value bubbles up to it's right position
def bubble_sort1(arr):
    while True:
        ix0 = 0
        ix1 = 1
        flag = 0
        while ix1 <= len(arr) - 1:
            if arr[ix0] > arr[ix1]:
                # arr[ix0], arr[ix1] = swap1(arr[ix0], arr[ix1])
                arr[ix0], arr[ix1] = swap2(arr[ix0], arr[ix1])

                flag = 1
            ix0 = ix0 + 1
            ix1 = ix1 + 1
        if flag == 0:
            break
    return arr

def bubble_sort2(arr):
    is_sorted = False
    while not is_sorted:
        ix0 = 0
        ix1 = 1
        is_sorted = True
        while ix1 <= len(arr) - 1:
            if arr[ix0] > arr[ix1]:
                arr[ix0], arr[ix1] = swap2(arr[ix0], arr[ix1])
                is_sorted = False
            ix0 += 1 
            ix1 += 1
    return arr

def bubble_sort3(arr):
    idx_sort = len(arr) - 1
    is_sorted = False
    while not is_sorted:
        is_sorted = True
        for i in range(idx_sort):
            if arr[i] > arr[i+1]:
                arr[i], arr[i+1] = arr[i+1], arr[i]
                is_sorted = False
        idx_sort -= 1 # in each pass the biggest unsorted value is in the right most place, hence -= 3
    return arr

if __name__ == "__main__":
    arr1 = [1, 3, 2, 7, 5, 10]
    arr2 = [1, 2, 3, 7, 8, 10]

    print(bubble_sort1(arr1))
    print(bubble_sort2(arr1))
    print(bubble_sort3(arr1))

    print(bubble_sort1(arr2))
    print(bubble_sort2(arr2))
    print(bubble_sort3(arr2))
