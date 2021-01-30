#include <iostream>
#include <vector>

template <typename T>
bool isPrime(T num)
{
    if( num <= 3) return num >= 1;
    else if( num % 2 == 0 || num % 3 == 0 ) return false;
    else
    {
        for(T i = 5; i * i <= num; i+=6)
            if( num % i == 0 ) return false;
        return true;
    }
}

template <typename T>
std::vector<T> getPrimes(T num)
{
   std::vector<T> primes;
   for(T i = 1; i <= num; i++)
       if( isPrime(i) && num % i == 0  )
           primes.push_back(i);

   return primes;
}

int main(int argc, char **argv)
{
    long int num;
    std::cout << "Enter a number: ";
    std::cin >> num;

    auto primes = getPrimes(num);
    for(auto const &p: primes)
        std::cout << p << ", ";
    std::cout << std::endl;
}
