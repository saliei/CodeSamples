#include <iostream>

/**
 * minimum function with any number of arguments.
 */

template <typename T>
T minimum(T const a, T const b)
{
    return a < b ? a : b;
}
template <typename T, typename...Args>
T minimum(T a, Args... args)
{
    return minimum(a, minimum(args...));
}

template <class Compare, typename T>
T minimumc(Compare comp, T const a, T const b)
{
    return comp(a, b) ? a: b;
}

template<class Compare, typename T, typename... Args>
T minimumc(Compare comp, T a, Args... args)
{
    return minimumc(comp, a, minimumc(comp, args...));
}

int main(int argc, char **argv)
{
    auto result = minimum(5, 2, 3, 1);
    std::cout << "result: " << result << std::endl;

    auto x = minimumc(std::less<>(), 3, 2, 5);
    std::cout << "result: " << x << std::endl;

    return 0;

}
