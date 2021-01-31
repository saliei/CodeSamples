#include <iostream>
#include <string>
#include <numeric>

bool is_isbn(std::string_view isbn )
{
    bool valid = false;

    if( isbn.size() == 10 && 
     std::count_if(isbn.begin(), isbn.end(), [](unsigned char c) { return std::isdigit(c); }) == 10 )
    {
        auto w = 10;
        auto sum = std::accumulate(isbn.begin(), isbn.end(), 0, 
                [&w](int const tot, char const c){
                return tot + w-- * (c - '0');
                });
        valid = !(sum % 11);
    }

    return valid;
}

int main(int argc, char **argv)
{
    std::string_view str = "1111111112";

    auto val = is_isbn(str);
    std::cout << val << std::endl;
}
