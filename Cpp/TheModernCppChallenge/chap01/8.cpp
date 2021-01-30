#include <iostream>
#include <vector>
#include <cmath>

std::vector<int> getDigits(int num)
{
    std::vector<int> digits;
    while(num > 9)
    {
        int n = num / 10;
        int d = num - n * 10;
        num = n;
        digits.push_back(d);
    }

    digits.push_back(num);

    return digits;
}

int main(int argc, char **argv)
{
    int num_digits;
    for(int i = 100; i < 1000; i++)
    {
        auto digits = getDigits(i);
        num_digits = digits.size();
        //std::cout << i << ": ";
        //for(auto const  &d: digits)
            //std::cout  << d << ", ";
        //std::cout << std::endl;
        int sum = 0;
        for(auto const &d: digits)
            sum += pow(d, num_digits);
        if( sum == i )
            std::cout << i << " is an Armstrong number." << std::endl;
    }

    return 0;
}
