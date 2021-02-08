#include <iostream>
#include <sstream>
#include <iomanip>

int main(int argc, char **argv)
{
    std::cout << "This is from std::cout" << std::endl; 

    std::ostringstream oss;
    oss << "This is from string stream" << std::endl;
    std::cout << oss.str();

    std::string strvar = "This is from string";

    oss << strvar << std::endl;
    std::cout << oss.str();

    oss.write("This is from write", 15);
    std::cout << oss.str() << std::endl;

    oss << std::setw(20) << std::right << "Formatted.";
    oss << std::fixed << std::setprecision(5) << 4.9872345 << std::endl;
    std::cout << oss.str() << std::endl;



    return 0;
}
