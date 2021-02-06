#include <iostream>
#include <string>

template <typename T>
T adder(T v)
{
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    return v;
}

template<typename T, typename... Args>
T adder(T first, Args... args)
{
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    return first + adder(args...);
}

template <typename T>
bool pair_compare(T a, T b)
{
    return a == b;
}

template <typename T, typename... Args>
bool pair_compare(T a, T b, Args... args)
{
    return a == b && pair_compare(args...);
}

int main(int argc, char **argv)
{
    long sum1 = adder(4, 6, 3);
    std::string s1 = "xx", s2 = "aa", s3 = "bb", s4 = "cc";
    std::string sum2 = adder(s1, s2, s3, s4);

    std::cout << "sum1: " << sum1 << std::endl;
    std::cout << "sum2: " << sum2 << std::endl;

    auto result = pair_compare(1.5, 1.5, 2, 2, 5, 5);
    std::cout << std::boolalpha << result << std::endl;

    return 0;
}

