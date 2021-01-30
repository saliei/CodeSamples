#include <iostream>
#include <type_traits>

void foo(char *) { std::cout << "foo(char *) was called." << std::endl; }
void foo(int  i) { std::cout << "foo(int  i) was called." << std::endl; }

int main(int argc, char **argv)
{
    if(std::is_same<decltype(NULL), decltype(0)>::value)
        std::cout << "NULL == 0" << std::endl;
    
    if(std::is_same<decltype(NULL), decltype((void*)0)>::value)
        std::cout << "NULL == (void *)0" << std::endl;

    if(std::is_same<decltype(NULL), std::nullptr_t>::value)
        std::cout << "NULL == nullptr" << std::endl;

    if(std::is_same<decltype(nullptr), std::nullptr_t>::value)
        std::cout << "nullptr is of type nullptr_t" << std::endl;

    foo(0);
    //foo(NULL); // compiler would not know which function to call.
    // is it 0 or a pointer?!
    // C++ does not allow to implicitly convert `void *` to other types.
    foo(nullptr);

    return 0;
}


