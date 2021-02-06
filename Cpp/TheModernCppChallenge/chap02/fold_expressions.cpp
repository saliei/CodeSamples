#include <iostream>

template<typename... Args>
auto avg(Args... args)
{
    return (args + ...) / sizeof...(args);
}

int main(int argc, char **argv)
{
    std::cout << avg(1.0, 2.0, 5.0, 2.0) << std::endl; 
}
