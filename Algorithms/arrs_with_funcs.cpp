#include <iostream>

using namespace std;

// static variables have scope throughout the lifetime of
// the program, so we can return static arrays from functions
int *fillArrayStatic()
{
    static int arr[3];
    arr[0] = 7;
    arr[1] = 8;
    arr[2] = 9;

    return arr;
}

// allocate dynamically the array with new operator to 
// insure that arr will be in scope by the time it will be reurned 
// from the function, as arr is dynamic it will be in scope unless
// we 'delete' it.
int *fillArrayDynamic()
{
    int *arr = new int[3];
    arr[0]=4;
    arr[1]=5;
    arr[2]=6;
    
    return arr;
}

int *fillArray()
{
    int arr[3] = {5, 6, 7};
//    return arr;
}

void printArray(int array[], int size)
{
    for(int i=0; i < size; ++i)
            cout << array[i] << "\t";
    cout << endl;
}

int main(int argc, char **argv)
{
    int arr[3] = {1, 2, 3};
    cout << "arr: " << arr << endl;
    cout << "&arr[0]: " << &arr[0] << endl;
    cout << "*arr: " << *arr << endl;
    cout << "*(&arr[0]): " << *(&arr[0]) << endl;
    cout << "*(arr+1): " << *(arr+1) << endl;
    
    cout << "============================" << endl;

    printArray(arr, 3);
//    Address binary error!
//    local variable returned from function and we dont know
//    if it will be in the scope by the time its returned.
//    int *arrPtr1 = fillArray();
//    printArray(arrPtr1, 3);
       
    cout << "============================" << endl;

    int *arrPtr1 = fillArrayDynamic();
    printArray(arrPtr1, 3);

    cout << "============================" << endl;

    int *arrPtr2 = fillArrayStatic();
    printArray(arrPtr2, 3);

    return 0;
}
