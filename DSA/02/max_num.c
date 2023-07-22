#include <stdio.h>
#include <stdbool.h>

int max_num_n2(int* arr, size_t size) {
    for(int i = 0; i < size; i++) {
        bool imax = true;
        for(int j = 0; j < size; j++) {
            if((i != j) && (arr[j] > arr[i])) imax = false;
        }
        if(imax) return arr[i];
    }
}

int max_num_n1(int* arr, size_t size) {
    int max = arr[0];
    for(int i = 1; i < size; i++) {
        if(arr[i] > max) max = arr[i];
    }
    return max;
}

int main() {
    int arr[5] = {1, 3, 2, 5, 5};

    int max1 = max_num_n1(arr, 5);
    printf("max1: %d\n", max1);

    int max2 = max_num_n2(arr, 5);
    printf("max2: %d\n", max2);
   
    return 0;
}
