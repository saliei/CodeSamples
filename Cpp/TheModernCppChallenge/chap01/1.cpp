#include <iostream>

int is3Divisable(int num)
{
    int flag = 0;
    if(num % 3 == 0)
        flag = 1;

    return flag;
}

int is5Divisable(int num)
{
    int flag = 0;
    if(num % 5 == 0)
        flag = 1;

    return flag;
}

int main(int argc, char **argv)
{
    unsigned int N;
    unsigned long long sum = 0;

    std::cin >> N;

    if( N < 0 )
    {
        fprintf(stderr, "Number must be positive.");
        exit(EXIT_FAILURE);
    }

    fprintf(stdout, "You eneterd: %d\n", N);

    for(unsigned int i = 0; i < N; i++)
    {
        if(is3Divisable(i) && is5Divisable(i))
            sum += i;
    }

    fprintf(stdout, "Sum is: %lld\n", sum);

    return 0;

}
