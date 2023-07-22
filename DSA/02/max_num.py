def max_num_n2(arr):
    for i in range(len(arr)):
        imax = True
        for j in range(len(arr)):
            if arr[j] > arr[i]:
                imax = False
        if imax:
            return i

def max_num_n1(arr):
    for i in range(len(arr)-1):
        _max = i
        if arr[i+1] > arr[i]:
            _max = i+1
    return _max


if __name__ == "__main__":
    arr = [2, 3, 2, 1, 5]
    print(arr[max_num_n2(arr)])
    print(arr[max_num_n1(arr)])
