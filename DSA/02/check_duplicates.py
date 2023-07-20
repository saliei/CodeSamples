# switching time complexity for space complexity!
def check_duplicate_n1(arr):
    new_arr = [0] * (max(arr)+1)
    for i in range(len(arr)):
        if new_arr[arr[i]] == 1:
            return True
        else:
            new_arr[arr[i]] = 1
    return False

def check_duplicate_n2(arr):
    for i in range(len(arr)):
        for j in range(len(arr)):
            if (i != j) and (arr[i] == arr[j]):
                return True
    return False


if __name__ == "__main__":
    arr1 = [2, 2, 3, 4]
    arr2 = [1, 2, 3, 4]

    print(check_duplicate_n1(arr1))
    print(check_duplicate_n1(arr2))

    print(check_duplicate_n2(arr1))
    print(check_duplicate_n2(arr2))

    
