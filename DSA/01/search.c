#include <stdio.h>
#include <stdlib.h>

#define SIZE 5

int linear_search(int* arr, int val) {
    for(size_t i = 0; i < SIZE; i++) {
        if (arr[i] == val) return i;
    }
    return -1;
}

int linear_search_ordered(int* arr, int val) {
    for(size_t i = 0; i < SIZE; i++) {
        if(arr[i] == val) return i;
        else if(arr[i] > val) break;
    }
    return -1;
}

int binary_search_recr(int* arr, int val, int low, int high) {
    int mid = (low + high) / 2;

    if (high >= low) {
        if(arr[mid] == val) return mid;
        else if(arr[mid] > val) return binary_search_recr(arr, val, low, mid-1);
        else return binary_search_recr(arr, val, mid+1, high);
    }
    else return -1;
}

int binary_search_iter(int* arr, int val) {
    int low = 0;
    int high = (int)(SIZE-1);
    int mid = 0;

    while(high >= low) {
        mid = (low + high) / 2;
        if(arr[mid] == val) return mid;
        else if(arr[mid] > val) high = mid-1;
        else low = mid+1;
    }
    return -1;
}

int main() {
    
    int arr[SIZE] = {1, 3, 5, 8, 10};
    printf("size: %lu\n", sizeof(arr)/sizeof(arr[0])); // this works iff arr hasn't passed to a function

    int idx1 = linear_search(arr, 10);
    printf("idx1: %d\n", idx1);

    int idx2 = linear_search_ordered(arr, 10);
    printf("idx2: %d\n", idx2);

    int idx3 = binary_search_recr(arr, 10, 0, SIZE-1);
    printf("idx3: %d\n", idx3);

    int idx4 = binary_search_iter(arr, 10);
    printf("idx4: %d\n", idx4);

    return 0;
}
