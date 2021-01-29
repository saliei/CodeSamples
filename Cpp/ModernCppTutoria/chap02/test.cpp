#include <iostream>

int main()
{
    std::cout << std::is_same<decltype(0),        decltype(NULL)>::value << std::endl; //false
    std::cout << std::is_same<decltype((void*)0), decltype(NULL)>::value << std::endl; //true
    std::cout << std::is_same<std::nullptr_t,     decltype(NULL)>::value << std::endl; //true

    return 0;
}
