#include <iostream>
#include <string>

class Ipv4
{
    public:
        std::string ip;
        Ipv4(std::string i): ip(i) {}
};

int main(int argc, char **argv)
{
    std::string ip;
    std::cout << "Enter ip: ";
    std::cin >> ip;

    Ipv4 ipv4(ip);

    std::cout << "You have entered ip: " << ipv4.ip << std::endl;
}

