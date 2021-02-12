#include <iostream>
#include <cstdio>
#include <cstdlib>

/**
 * Use cases:
 * 1: functions as arguments to other function, e.g. sorting behaviour
 * 2: callback functions, e.g. function to run when an event is happend. e.g. create_button
 *    void create_button(int x, int y, const char *text, function callback_function)
 */

// qsort from Linux man pages
// void qsort(void *base, size_t nmemb, size_t size,
//                  int(*comapre)(const void *, const void *))

// using virtual functions instead of function pointers

class Sorter
{
    public:
        virtual int compare(const void *first, const void *second);
};

void cpp_qsort(void *base, size_t nmemb, size_t size, Sorter *compar);

//inside cpp_qsort, whenever a comparison is needed, compar->compare should be called. 
//For classes that override this virtual function, the sort routine will 
//get the new behavior of that function

class AscendSorter: public Sorter
{
    virtual int comapre(const void *first_arg, const void *second_arg)
    {
        int first = *(int*)first_arg;
        int second= *(int*)second_arg;
        if(first < second)
            return -1;
        else if(first == second)
            return 0;
        else
            return 1;
    }
};

void my_int_func(int x)
{
    printf("%d\n", x);
}

int int_sorter(const void *first_arg, const void *second_arg)
{
    int first = *(int*)(first_arg);
    int second= *(int*)(second_arg);

    if(first < second)
        return -1;
    else if(first == second)
        return 0;
    else
        return 1;
}

int main(int argc, char **argv)
{
    void (*foo)(int);
    foo = &my_int_func;
    
    // calling the func pointer would automatically derefrence it.
    foo(2);
    // or you can do it by yourself
    (*foo)(2);

    std::cout << std::endl;

    int arr[10];
    int i;

    for(i = 0; i < 10; i++)
        arr[i] = 10 - i;

    qsort(arr, 10, sizeof(int), int_sorter);

    for(i = 0; i < 10; i++)
        printf("%d\n", arr[i]);

    return 0;


}
