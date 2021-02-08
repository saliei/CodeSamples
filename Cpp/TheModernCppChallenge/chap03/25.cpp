#include <iostream>
#include <string>
#include <algorithm>
#include <cctype>
#include <assert.h>

template<class Elem>
using tstring = std::basic_string<Elem, std::char_traits<Elem>, std::allocator<Elem>>;

template<class Elem>
using tstringstream = std::basic_string<Elem, std::char_traits<Elem>, std::allocator<Elem>>;

template<class Elem>
tstring<Elem> capitalize(tstring<Elem> const &text)
{
    tstringstream<Elem> result;
    bool newWord = true;
    for(auto const &ch: text)
    {
        newWord = newWord || std::ispunct(ch) || std::isspace(ch);
        if(std::isalpha(ch))
        {
            if(newWord)
            {
                result << static_cast<Elem>(std::toupper(ch));
                newWord = false;
            }
            else
                result << static_cast<Elem>(std::tolower(ch));
        }
        else result << ch;
        
    }
}

int main(int argc, char **argv)
{
    using namespace std::string_literals;
    assert("The C++ Challenger"s == capitalize("the c++ challenger"s));
    assert("This Is An Example, Should Work!"s == capitalize("thIs Is an eXamPle, SHOULD woRk!"s));

    return 0;

}








