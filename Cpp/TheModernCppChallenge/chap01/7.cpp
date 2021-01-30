#include <iostream>
#include <vector>

#define LIMIT 1000000

std::vector<long int> getDivs(int num)
{
    std::vector<long int> divs;

    for(long int i = 1; i < num; i++)
        if( num % i == 0 )
            divs.push_back(i);

    return divs;

}

int main(int argc, char **argv)
{
    for(long int i = 1; i < LIMIT; i++)
    {
        auto divs1 = getDivs(i);
        long int sum1 = 0;
        for( auto const &x: divs1)
            sum1 += x;

        auto divs2 = getDivs(sum1);
        long int sum2 = 0;
        for(auto const &x: divs2)
            sum2 += x;
        if(sum2 == i && i < sum1)
            std::cout << "(" << i << ", " << sum1 << ")" << std::endl;
    }

    return 0;
}
