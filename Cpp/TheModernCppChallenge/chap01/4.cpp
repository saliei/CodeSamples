#include <iostream>

int isPrime(unsigned long num)
{
    int flag = 0;
    if(num == 1 || num == 2)
        return 1;
    
    else
    {
        for(int i = 2; i <= num/2; i++)
        {
            if( num % i == 0 )
            {
                flag = 0;
                break;
            }
            else flag = 1;
        }
    }

    return flag;
}

int main(int argc, char **argv)
{
    unsigned long N;
    unsigned long b = 1;

    std::cout << "Enter a number: " << std::endl;

    std::cin >> N;

    for( int i = 2; i <= N; i++)
    {
        if (isPrime(i) && i > b)
            b = i;
    }

    std::cout << "biggest prime: " << b << std::endl;

    return 0;
}
