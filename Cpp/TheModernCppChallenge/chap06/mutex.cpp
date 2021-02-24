#include <iostream>
#include <thread>
#include <mutex>

std::mutex mt;
int i = 0;

void make_a_call()
{
    mt.lock();

    std::cout << i << " hello" << std::endl;
    i++;

    mt.unlock();
}

int main(int argc, char **argv)
{
    std::thread man1(make_a_call);

    std::thread man2(make_a_call);

    std::thread man3(make_a_call);

    man1.join();
    man2.join();
    man3.join();

    return 0;
}
