#include <stdio.h>
#include <stdlib.h>

#define SIZE 5

void selection_sort(int* arr) {
    int i, j;
    for(i = 0; i < SIZE-1; i++) {
        int min_idx = i;
        for(j = i+1; j < SIZE; j++) {
            if(arr[j] < arr[i]) {
                min_idx = j;
            }
        }
        if(min_idx != i) {
            int tmp = arr[i];
            arr[i] = arr[min_idx];
            arr[min_idx] = tmp;
        }
    }
}

int main() {
    int arr1[SIZE] = {1, 4, 3, 2, 5};
    int arr2[SIZE] = {1, 2, 3, 4, 5};

    selection_sort(arr1);
    selection_sort(arr2);

    for(int i = 0; i < SIZE; i++) printf("%d ", arr1[i]);
    printf("\n");
    for(int i = 0; i < SIZE; i++) printf("%d ", arr2[i]);
    printf("\n");
}
