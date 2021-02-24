#include <iostream>
#include <algorithm>
#include <vector>
#include <map>

template<typename T>
std::vector<std::pair<T, size_t>> find_most_frequent(const std::vector<T>& range)
{
    std::map<T, size_t> counts;

    for(const auto& e: range) counts[e]++;

    auto maxelem = std::max_element(
            std::cbegin(counts), std::cend(counts),
            [](const auto& e1, const auto& e2){
            return e1.second < e2.second;
            }
            );

    std::vector<std::pair<T, size_t>> result;

    std::copy_if(
            std::begin(counts), std::end(counts),
            std::back_inserter(result),
            [maxelem](const auto& kvp){
            return kvp.second == maxelem->second;
            }
            );

    return result;
}

int main(int argc, char **argv)
{
    auto range = std::vector<int>{1,1,3,5,8,13,3,5,8,8,5};
    auto result = find_most_frequent(range);

    for(const auto& e: result) std::cout << e.first << ": " << e.second << std::endl;
    

    return 0;
}
