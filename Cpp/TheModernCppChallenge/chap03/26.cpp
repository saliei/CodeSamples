#include <iostream>
#include <sstream>
#include <string>
#include <assert.h>
#include <iterator>
#include <vector>

template<typename Iter>
std::string join_strings(Iter begin, Iter end, const char * const seperator)
{
    std::ostringstream os;
    std::copy(begin, end-1, std::ostream_iterator<std::string>(os, seperator));
    os << *(end-1);
    return os.str();
}

template<typename Cont>
std::string join_strings(const Cont &c, const char * const seperator)
{
    if(c.size() == 0) return std::string();

    return join_strings(std::begin(c), std::end(c), seperator);
}

int main(int argc, char **argv)
{   
    using namespace std::string_literals;

    std::vector<std::string> vec1{"this", "is", "an", "example"};
    std::vector<std::string> vec2{"example"};
    std::vector<std::string> vec3{};

    assert(join_strings(vec1, " ") == "this is an example"s);
    assert(join_strings(vec2, " ") == "example"s);
    assert(join_strings(vec3, " ") == ""s);

    return 0;

}
