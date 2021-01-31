#include <iostream>
#include <random>
#include <tuple>

// get a random number between a and b, n times
auto get_random_doubles(const double a, const double b, const size_t n = 1000000)
{
    std::vector<std::tuple<double, double>> rands;
    double randx, randy;

    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<double> distribution(a, b);
    
    for(size_t i = 0; i < n; i++) {
        randx = distribution(generator);
        randy = distribution(generator);
        
        auto tuple = std::make_tuple(std::move(randx), std::move(randy));
        rands.push_back(tuple);
    }

    // another way from the `The Modern C++ Challenge`
/*
 *    std::random_device rd;
 *    auto seed_data = std::array<int, std::mt19937::state_size>{};
 *    std::generate(std::begin(seed_data), std::end(see_data), std::ref(rd));
 *    std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
 *    auto eng  = std::mt19937{seq};
 *    auto dist = std::uniform_real_distribution<>{0, 1};
 *
 *    auto x = dist(eng);
 *    auto y = dist(eng);
 */


    return rands;
}

int main(int argc, char **argv)
{

    auto rands = get_random_doubles(0.0, 1.0);
    
    // std::scientific
    /*
     *std::cout << std::fixed; std::cout.precision(2);
     *for( auto const &x: rands )
     *    std::cout << x << ", ";
     *std::cout << std::endl;
     */

    /*
     *std::cout << std::fixed; std::cout.precision(2);
     *for( auto const &x: rands )
     *    std::cout << "(" << std::get<0>(x) << ", " << std::get<1>(x) << ")"<< std::endl;
     */
     
    // circle counts
    long int ccount = 0;
    // rectangle couns
    long int rcount = rands.size();
    for( const auto &r: rands )
    {
        auto x = std::get<0>(r);
        auto y = std::get<1>(r);
        auto r2 = x * x + y * y;
        if( r2 <= 1.0 ) ccount += 1;
    }
        
    double ratio = ccount / (double)rcount;

    std::cout << std::fixed; std::cout.precision(4);
    std::cout << "PI: " << 4 * ratio << std::endl;

    
   

    return 0;
}
