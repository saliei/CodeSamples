#include <iostream>
#include <vector>


template<typename InputItr, typename OutputItr>
void pairwise(InputItr begin, InputItr end, OutputItr result)
{
    auto itr = begin;
    while(itr != end)
    {
        auto v1 = *itr++; if (itr == end) break;
        auto v2 = *itr++;
        result++ = std::make_pair(v1, v2);
    }
}


template<typename T>
std::vector<std::pair<T, T>> pairwise(const std::vector<T> &range)
{
    std::vector<std::pair<T, T>> result;

    pairwise(std::begin(range), std::end(range), std::back_inserter(result));
    return result;
}

template<typename T>
std::ostream& operator<<(std::ostream &os ,const std::vector<std::pair<T, T>> &pairs)
{
    for(const auto &p: pairs)
        os << "{" << p.first << ", " << p.second << "}" << std::endl;

    return os;
}

int main(int argc, char **argv)
{
    std::vector<int> range{1, 3, 4, 5, 3, 5, 6};
    auto result = pairwise(range);

    for(const auto &p: result)
        std::cout << "{" << p.first << ", " << p.second << "}" << std::endl;

    std::cout << std::endl;

    std::cout << result;

   return 0;
}
