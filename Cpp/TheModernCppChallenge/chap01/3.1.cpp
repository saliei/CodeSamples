#include <iostream>
#include <numeric>
#include <vector>

/** 
 * lcm(a, b) = multiply(a, b) / gcd(a, b)
 */

unsigned int lcm(const unsigned int a, const unsigned int b)
{
    unsigned int h = std::gcd(a, b);

    return h ? (a * (b / h)) : 0;
}

template<class InputItr>
unsigned int lcmr(InputItr first, InputItr last)
{
    return std::accumulate(first, last, 1, lcm);
}

int main(int argc, char **argv)
{
    int num;
    
    //std::vector<int> nums = {6, 14, 5};
    std::vector<int> nums;
    while(std::cin >> num)
        nums.push_back(num);

    unsigned int lm = lcmr(nums.begin(), nums.end());

    std::cout << "lcm: " << lm << std::endl;

    return 0;

}
