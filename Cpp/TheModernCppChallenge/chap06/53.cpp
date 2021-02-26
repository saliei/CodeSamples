#include <iostream>
#include <numeric>
#include <iomanip>
#include <string>
#include <vector>

struct movie
{
    int id;
    std::string title;
    std::vector<int> ratings;
};

double truncated_mean(std::vector<int> values, double percentage)
{

    std::sort(std::begin(values), std::end(values), [](int a, int b){
            return a < b;
            });
   auto count = static_cast<size_t>(values.size() * percentage);

   values.erase(std::begin(values), std::begin(values) + count);
   values.erase(std::end(values) - count, std::end(values));
   auto total = std::accumulate(std::cbegin(values), std::cend(values), 0ull, 
           [](auto const sum, auto const value){
           return sum + value;
           });

   return static_cast<double>(total) / values.size();
}

void print_ratings(std::vector<movie> const &movies)
{
    for(auto const &m: movies)
    {
        std::cout << m.title << " : " << std::fixed << std::setprecision(1) 
            << truncated_mean(m.ratings, 0.05) << std::endl;
    }
}


int main(int argc, char **argv)
{

    std::vector<movie> movies{
        {101, "Law Abiding Citizen", {10, 9, 10, 9, 9, 8, 7, 10, 5, 9, 9, 8}},
        {102, "Interstellar", {10, 5, 7, 8, 9, 8, 9, 10, 10, 5, 9, 8, 10}},
        {103, "Gladiator", {10, 10, 10, 9, 3, 8, 8, 9, 6, 4, 7, 10 }}
    };

    print_ratings(movies);

    return 0;

}
