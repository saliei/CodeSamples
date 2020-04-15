#include <iostream>
#include <vector>
#include <algorithm>

template <typename Iter, typename Cmp>
void BubbleSort(Iter first, Iter last, Cmp cmp)
{
    bool done = false;
    while(!done)
    {
        done = true;
        for(auto it=first; it != last; ++it)
            if(!cmp(*it, *(it+1)))
            {
                std::swap(*it, *(it+1));
                done = false;
            }
    }
}

template <typename T>
bool cmp(T x, T y) { return x < y; }

int main()
{
    std::vector<int> vec{3, 1, 3, 2, 6, 4, 5, 7, 8, 6};
    BubbleSort(vec.begin(), vec.end(), cmp<int>);
    for(auto i: vec) std::cout << i << std::endl;
    return 0;
}
