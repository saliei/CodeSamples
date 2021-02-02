#include <iostream>
#include <vector>
#include <string>
#include <sstream>

int main(int argc, char **argv)
{
    std::string str{"Hello from the dark side"};
    std::cout << str << std::endl;
    
    std::stringstream sstr(str); // make a stream out of str, it's like std::cin or std::cout
    std::string tmp; // tmp value to hold each word.
    std::vector<std::string> words;

    while( sstr >> tmp )
    {
        words.push_back(tmp);
    }

    for(const auto &word: words)
        std::cout << word << std::endl;

    int decimal = 61;
    std::stringstream sstr1;
    sstr1 << std::hex << decimal;
    std::string res = sstr1.str();
    std::cout <<  "The hexadeciamal of " << decimal << " is: " << res << std::endl;

    return 0;
}
