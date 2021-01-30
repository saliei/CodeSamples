#include <iostream>
#include <vector>

std::vector<int> getDivs(long int num)
{
    std::vector<int> divs;

    for(int i = 1; i < num; i++)
        if( num % i == 0 )
            divs.push_back(i);

    return divs;
}

int main(int argc, char **argv)
{   
    long int num;
    std::cout << "Enter number: ";
    std::cin >> num;

    std::vector<int> divs = getDivs(num);
    
    long sum = 0;
    for(auto const &x: divs)
        sum += x;

    std::cout << "sum: "<< sum << std::endl;

    if( sum > num ) std::cout << "abundance: " << sum - num << std::endl;

    return 0;

}
