#include <iostream>
#include <tuple>

// return multiple values form a function


std::tuple<int, double, std::string> f() { return std::make_tuple(1, 1.0, "1"); }

template <typename T> std::string type_name();

int main(int argc, char **argv)
{
    auto [x, y, z] = f();

    std::cout << x << ", " << y << ", " << z << std::endl;

    std::cout << typeid(x).name() << std::endl;
    std::cout << typeid(y).name() << std::endl;
    std::cout << typeid(z).name() << std::endl;

    //std::cout << type_name<decltype(x)>() << '\n';

    return 0;
}
