#include <iostream>
#include <vector>

template<typename InputItr1, typename InputItr2, typename OutputItr>
void zip(InputItr1 begin1, InputItr1 end1, InputItr2 begin2, InputItr2 end2, OutputItr result)
{
    auto it1 = begin1;
    auto it2 = begin2;
    while(it1 != end1 && it2 != end2)
        result++ = std::make_pair(*it1++, *it2++);

}


template<typename T, typename U>
std::vector<std::pair<T, U>> zip(const std::vector<T> &v1, const std::vector<U> &v2)
{
    std::vector<std::pair<T, U>> result;
    zip(std::begin(v1), std::end(v1), 
        std::begin(v2), std::end(v2), 
        std::back_inserter(result));

   return result;
}


template<typename T, typename U>
std::ostream& operator<<(std::ostream& os, const std::vector<std::pair<T, U>>& result)
{
    for(const auto &p: result)
    {
        os << "{" << p.first << ", " << p.second << "}" << std::endl;
    }

    return os;
}

int main(int argc, char **argv)
{
    std::vector<int> v1{1, 4, 2, 4, 3, 5, 9, 7};
    std::vector<double> v2{1.4, 3.2, 4.1, 2.0};

    std::vector<std::pair<int, double>> result = zip(v1, v2);

    std::cout << result;
}
