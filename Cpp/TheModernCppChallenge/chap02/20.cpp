#include <iostream>
#include <vector>
#include <array>
#include <list>

/**
 * Contains any, all, or none.
 * */
template<class Container, class Type>
bool contains(Container const &cont, Type const &value)
{
    return std::end(cont) != std::find(std::begin(cont), std::end(cont), value);
}

template<class Container, class... Args>
bool contains_any(Container const &cont, Args &&... value)
{
    return (... || contains(cont, value));
}

template<class Container, class... Args>
bool contains_all(Container const &cont, Args &&... value)
{
    return (... && contains(cont, value));
}

template <class Container, class... Args>
bool contains_none(Container const &cont, Args &&... value)
{
    return !contains_any(cont, std::forward<Args>(value)...);
}

int main(int argc, char **argv)
{
    std::vector<int> vec{3, 2, 1, 5};
    std::array<float, 3> arr{ {4.0, 7.0, 3.0} };
    std::list<int> lis{ 8, 9, 10 };

    std::cout << std::boolalpha <<"vec contains 3: " << contains(vec, 3) << std::endl;
    std::cout << std::boolalpha <<"vec contains 4: " << contains(vec, 4) << std::endl;

    std::cout << std::boolalpha << "arr contains any 4.0 or 5.0: " << contains_any(arr, 4.0, 5.0) << std::endl;
    std::cout << std::boolalpha << "arr contains all 4.0 or 5.0: " << contains_all(arr, 4.0, 5.0) << std::endl;

    std::cout << std::boolalpha << "lis contains none 8 and 9: " << contains_none(lis, 8, 9) << std::endl;
    std::cout << std::boolalpha << "lis contains all 8 and 9: " << contains_all(lis, 8, 9) << std::endl;

    return 0;
}
