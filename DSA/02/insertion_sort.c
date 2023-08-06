#include <stdio.h>
#include <stdlib.h>

#define SIZE 5

void insertion_sort1(int* arr) {
    for(int i = 1; i < SIZE; i++) {
        for(int j = i; j > 0 && arr[j-1] > arr[j]; j--) {
            int tmp = arr[j-1];
            arr[j-1] = arr[j];
            arr[j] = tmp;
        }
    }
}

void insertion_sort2(int* arr) {
    for(int i = 1; i < SIZE; i++) {
        int key = arr[i];
        for(int j = i; j > 0 && arr[j-1] > arr[j]; j--) {
            arr[j] = arr[j-1];
            arr[j] = key;
        }
    }
}

int main() {
    int arr1[SIZE] = {3, 2, 1, 5, 4};
    int arr2[SIZE] = {1, 2, 3, 4, 5};
    insertion_sort1(arr1);
    insertion_sort1(arr2);
    for(int i = 0; i < SIZE; i++) {
        printf("%d, %d\n", arr1[i], arr2[i]);
    }

    int arr3[SIZE] = {3, 2, 1, 5, 4};
    int arr4[SIZE] = {1, 2, 3, 4, 5};
    insertion_sort2(arr3);
    insertion_sort2(arr4);
    for(int i = 0; i < SIZE; i++) {
        printf("%d, %d\n", arr3[i], arr4[i]);
    }
}
