#include <iostream>
#include <vector>



template<class RandomIt>
RandomIt partition(RandomIt first, RandomIt last)
{
    auto pivot = *first;
    auto i = first + 1;
    auto j = last - 1;

    while( i <= j )
    {
        while(i <= j && *i <= pivot) i++;
        while(i <= j && *j > pivot) j--;
        if(i < j) std::iter_swap(i, j);
    }

    std::iter_swap(i - 1, first);

    return i - 1;
}

template<class RandomIt, class Compare>
RandomIt partition(RandomIt first, RandomIt last, Compare comp)
{
    auto pivot = *first;
    auto i = first + 1;
    auto j = last - 1;

    while(i <= j)
    {
        while(i <= j && comp(*i, pivot)) i++;
        while(i <= j && !comp(*j, pivot)) j--;
        if(i < j) std::iter_swap(i, j);
    }

    std::iter_swap(i-1, first);
    
    return i - 1;
}
