#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <sstream>
#include <assert.h>
#include <iomanip>

template<typename Iter>
std::string bytesToHexStr(Iter begin, Iter end, bool const uppercase = false)
{
    std::ostringstream oss;
    if(uppercase) oss.setf(std::ios_base::uppercase);
    for(; begin != end; ++begin)
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(*begin);

    return oss.str();
}

template<typename Cont>
std::string bytesToHexStr(Cont const &c, bool const uppercase = false)
{
    return bytesToHexStr(std::cbegin(c), std::cend(c), uppercase);
}


int main(int argc, char **argv)
{

    std::vector<unsigned char> vec{0xBA, 0xAD, 0xF0, 0x0D};
    std::array<unsigned char, 6> arr{ {1, 2, 3, 4, 5, 6} };
    unsigned char buf[5] = {0x11, 0x22, 0x33, 0x44, 0x55};

    assert(bytesToHexStr(vec, true) == "BAADF00D");
    assert(bytesToHexStr(arr, true) == "010203040506");
    assert(bytesToHexStr(buf, true) == "1122334455");

    assert(bytesToHexStr(vec) == "baadf00d");
    assert(bytesToHexStr(arr) == "010203040506");
    assert(bytesToHexStr(buf) == "1122334455");

    return 0;

}

