#include <iostream>
#include <bitset>

int main(int argc, char **argv)
{
    int a = 192; int b = -10; int c = a >> 3; 
    int d = a > 3;
    int e = (168 >> 16) & 0xFF;
    int f = static_cast<unsigned char>((68 >> 16) & 0xFF);

    std::cout << "a<8>  = " << std::bitset<8>(a) << std::endl;
    std::cout << "a<16> = " << std::bitset<16>(a) << std::endl;
    std::cout << "b = " << std::bitset<8>(b) << std::endl;
    std::cout << "c = " << std::bitset<8>(c) << std::endl;
    std::cout << "d = " << std::bitset<8>(d) << std::endl;
    std::cout << "e = " << std::bitset<32>(e) << std::endl;
    std::cout << "e = " << e << std::endl;
    std::cout << "f = " << f << std::endl;


    return 0;
}
