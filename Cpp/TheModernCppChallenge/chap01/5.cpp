#include <iostream>

bool isPrime(long int num)
{
    if (num <= 3) return num > 1;
    else if( num % 2 == 0 || num % 3 == 0 ) return false;
    else
    {
        for(int i = 5; i * i <= num; i += 6)
            if( num % i == 0 ) return false;
        return true;
    }
}

int main(int argc, char **argv)
{
    int lim;
    std::cout << "Enter a limit: ";
    std::cin >> lim;

    for(int i = 1; i+6 <= lim; i++)
        if( isPrime(i) && isPrime(i+6) ) 
            std::cout << "(" << i << ", " << i+6 << ")" << std::endl;

    return 0;
}
