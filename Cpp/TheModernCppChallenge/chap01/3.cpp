#include <iostream>
#include <vector>

int isPrime(int num)
{
    int flag = 0;
    if(num == 1 || num == 2)
        flag = 1;
    else
    {
        for(int i = num/2; i < num; i++)
        {
            if(num % i == 0)
                flag = 0;
            else
                flag = 1;
        }
    }

    return flag;
}

int main(int argc, char **argv)
{
    int N, M;

    std::cout << "Enter two number: \n";
    std::cin >> N;
    std::cin >> M;
    
    std::vector<int> nmuls;
    std::vector<int> mmuls;

    for(int i = 2; i < N; i++)
        if(N % i == 0 && isPrime(i))
            nmuls.push_back(i);
    for(int i = 2; i < M; i++)
        if(M % i == 0 && isPrime(i))
            mmuls.push_back(i);

    nmuls.insert(nmuls.end(), mmuls.begin(), mmuls.end());
    std::sort(nmuls.begin(), nmuls.end());
    nmuls.erase(std::unique(nmuls.begin(), nmuls.end()), nmuls.end());

    unsigned long mulp = 1;

    for(auto itr = nmuls.begin(); itr != nmuls.end(); itr++)
        mulp *= *itr;
    
    std::cout << "lcm is: " << mulp << std::endl;

    return 0;

}
