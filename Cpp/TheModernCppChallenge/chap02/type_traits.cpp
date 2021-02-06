#include <iostream>
#include <type_traits>

class Class {};

void algorithm_signed(int i) {std::cout << "signed is picked!" << std::endl;}
void algorithm_unsigned (unsigned u) { std::cout << "unsigned is picked!" << std::endl; }

template<typename T>
void algorithm(T t)
{
    if constexpr(std::is_signed<T>::value)
        algorithm_signed(t);
    else
        if constexpr(std::is_unsigned<T>::value)
            algorithm_unsigned(t);
    else
        static_assert(std::is_signed<T>::value || std::is_unsigned<T>::value, "Must be signed or unsigned!");

}

int main(int argc, char **argv)
{
    std::cout << std::is_floating_point<Class>::value << std::endl;
    std::cout << std::is_floating_point<float>::value << std::endl;
    std::cout << std::is_floating_point<int>::value << std::endl;

    algorithm(3);
    unsigned x = 3;
    algorithm(x);

    //algorithm("hello"); // static insertation failure!
}
