#include <iostream>
#include <vector>
#include <string>
#include <string_view>
#include <algorithm>


bool starts_with(std::string_view str, std::string_view prefix)
{
    return str.find(prefix) == 0;
}

template<typename InputItr>
std::vector<std::string> filter_numbers(InputItr begin, InputItr end, const std::string& country_code)
{
    std::vector<std::string> result;

    std::copy_if(
            begin, end,
            std::back_inserter(result),
            [country_code](const auto& number){
            return starts_with(number, country_code) || 
            starts_with(number, "+" + country_code);
            }
            );

    return result;
}

std::vector<std::string> filter_numbers(
        const std::vector<std::string>& numbers,
        const std::string& country_code
        )
{
    return filter_numbers(
            std::begin(numbers), std::end(numbers),
            country_code
            );
}

int main(int argc, char **argv)
{
    std::vector<std::string> numbers{
      "+40744909080",
      "44 7520 112233",
      "+44 7555 123456",
      "40 7200 123456",
      "7555 123456"
    };
    

    auto result = filter_numbers(numbers, "44");

    for(const auto& number: result) std::cout << number << std::endl;

    return 0;

}
