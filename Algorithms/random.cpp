#include <iostream>
#include <random>
#include <vector>
#include <functional>


int main()
{
    double dlb = 0, dub = 100;
    int ilb = 0, iub = 100;
    std::uniform_real_distribution<double> dunif(dlb, dub);
    std::uniform_int_distribution<int> iunif(ilb, iub);
    std::default_random_engine eng;
    auto idist = std::bind(iunif, eng);
    auto ddist = std::bind(dunif, eng);
    for(int i=0; i<20; ++i)
        std::cout << idist() << std::endl;
    std::cout << "=============" << std::endl;
    for(int i=0; i<20; ++i)
        std::cout << ddist() << std::endl;

    std::vector<int> vec(10);
   
}
