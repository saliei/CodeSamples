#include <iostream>
#include <vector>
#include <string>
#include <string_view>

#define ASIO_STANDALONE
#include <asio.hpp>

std::vector<std::string> get_ip_address(std::string_view hostname)
{
    std::vector<std::string> ips;

    try
    {
        asio::io_context context;
        asio::ip::tcp::resolver resolver(context);
        auto endpoints = resolver.resolve(asio::ip::tcp::v4(), hostname.data(), "");

        for(auto e = endpoints.begin(); e != endpoints.end(); ++e)
            ips.push_back(((asio::ip::tcp::endpoint)*e).address().to_string());
    }
    catch(std::exception const &e)
    {
        std::cerr << "exception: " << e.what() << std::endl;
    }

    return ips;
}

int main(int argc, char **argv)
{
    std::string hostname;
    std::cout << "Enter hostname:\n> ";
    std::cin >> hostname;
    
    auto ips = get_ip_address(hostname);

    for(auto const &ip: ips)
        std::cout << ip << std::endl;

    return 0;
}
