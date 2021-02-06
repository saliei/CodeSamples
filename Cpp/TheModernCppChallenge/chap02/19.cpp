#include <iostream>
#include <vector>
#include <list>
#include <iterator>

template<typename Container, typename... Args>
void push_back(Container& cont, Args&&... args)
{
    (cont.push_back(args), ...);
}


int main(int argc, char **argv)
{
    std::vector<int> vec{1, 3, 2, 5};    
    push_back(vec, 6, 7, 9);
    std::copy(std::begin(vec), std::end(vec), std::ostream_iterator<int>(std::cout, ", "));
    std::cout << std::endl;

    std::list<int>lis;
    push_back(lis, 4, 2, 3, 1);
    std::copy(std::begin(lis), std::end(lis), std::ostream_iterator<int>(std::cout, ", "));
    std::cout << std::endl;

    return 0;
}
