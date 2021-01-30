#include <iostream>


bool isPrime(int num)
{
    if( num <= 3 ) return num > 1;
    else if( num % 2 == 0 || num % 3 == 0 ) return false;
    else
    {
        for(int i = 5; i * i <= num; i += 6 )
            if ( num % i == 0 || num % (i+2) == 0 )
                return false;
        return true;
    }
}

int main(int argc, char **argv)
{
    int num;

    std::cout << "Enter number: ";
    std::cin >> num;

    for(int i = num; i > 1; i-- )
        if( isPrime(i) )
        {
            std::cout << "Greatest prime: " << i << std::endl;
            return 0;
        }
}
