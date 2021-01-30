#include <iostream>

#define LEN 10

// length of an array should be `constexpr`

int len_foo()
{
    int i = 2;
    return i;
}

constexpr int len_foo_constexpr()
{
    return 5;
}

constexpr int fibonacci(const int n)
{
    return n == 1 || n == 2 ? 1 : fibonacci(n-1) + fibonacci(n-2);
}



int main(int argc, char **argv)
{
    char arr_1[10];                 //legal
    char arr_2[LEN];                //legal

    int len = 10;
    char arr_3[len];                //illegal

    const int len_2 = len + 1;
    constexpr int len_2_constexpr = 1 + 2 + 3;
    
    // char arr_4[len_2];               //illegal, but Ok for most compilers
    char arr_5[len_2_constexpr];        //legal

    // char arr_6[len_foo() + 5];       //illegal, before C++98 compiler does not know that `len_foo()` returns a const at runtime.
    char arr_7[len_foo_constexpr() + 5];

    std::cout << fibonacci(10) << std::endl;

    return 0;
}
