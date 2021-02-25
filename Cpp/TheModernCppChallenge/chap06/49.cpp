#include <iostream>
#include <map>
#include <numeric>
#include <iomanip>
#include <string>
#include <string_view>


std::map<char, double> analyze_text(std::string_view text)
{
    std::map<char, double> frequencies;
    for(char ch='a'; ch <= 'z'; ch++)
        frequencies[ch] = 0;

    for(auto ch: text)
    {
        if(isalpha(ch))
            frequencies[tolower(ch)]++;
    }

    auto total = std::accumulate(
            std::cbegin(frequencies), std::cend(frequencies), 0ull,
            [](const auto sum, const auto& kvp){
            return sum + static_cast<unsigned long long>(kvp.second);
            }
            );

    std::for_each(
            std::begin(frequencies), std::end(frequencies),
            [total](auto& kvp){
            kvp.second = (100.0 * kvp.second) / total;
            }
            );

    return frequencies;
}

int main(int argc, char **argv)
{
    auto result = analyze_text(R"(This is a simple text to analyze and find the 
                                  the frequencies for.)");

    for(const auto& kvp: result)
    {
        std::cout << kvp.first << " : " << std::fixed << std::setw(5) 
            << std::setfill(' ') << std::setprecision(2) << kvp.second << std::endl;
    }

    return 0;
}
