#include <iostream>
#include <numeric>

/**
 * gcd(a, 0) = a;
 * gcd(a, b) = gcd(b, a mode b)
 */

unsigned int gcd_r(const unsigned int a, const unsigned int b)
{
    return  b == 0 ? a : gcd_r(b, a % b);
}

unsigned int gcd_nr(unsigned int a, unsigned int b)
{
    while( b != 0 )
    {
        unsigned int r = a % b;
        a = b;
        b = r;
    }

    return a;
}

int main()
{
    int N, M;

    std::cout << "Enter two number: \n";
    std::cin >> N;
    std::cin >> M;
    
    unsigned int gcd1  = gcd_r(N, M);
    unsigned int gcd2  = gcd_nr(N, M);

    std::cout << "gcd_r: "  << gcd1  << std::endl;
    std::cout << "gcd_nr: " << gcd2 << std::endl;
    std::cout << "gcd: "    << std::gcd(N, M) << std::endl;

    return 0;

}
