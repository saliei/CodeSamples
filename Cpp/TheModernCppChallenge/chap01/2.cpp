#include <iostream>
#include <vector>


// handle special case when one of them is zero

int main(int argc, char **argv)
{
    int N, M;
    std::vector<int> ndivs, mdivs, comms;

    std::cout << "Enter two number: \n";
    std::cin >> N;
    std::cin >> M;

    for(int i = 1; i <= N; i++)
        if( N % i == 0 )
            ndivs.push_back(i);

    for(int i = 1; i <= M; i++)
        if( M % i == 0 )
            mdivs.push_back(i);

    for(auto itr = ndivs.begin(); itr != ndivs.end(); itr++)
        if(std::find(mdivs.begin(), mdivs.end(), *itr) != mdivs.end())
            comms.push_back(*itr);

    std::cout <<  "Greatest common divisor: " << comms[comms.size()-1] << std::endl;

    return 0;

}
