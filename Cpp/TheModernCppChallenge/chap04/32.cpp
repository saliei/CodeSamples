#include <iostream>
#include <string>

unsigned int number_of_digits(const unsigned int i)
{
    return i > 0 ? i+1: 1;
}

void print_pascal_triangle(const int n)
{
    auto x = 1;

}

int main(int argc, char **argv)
{
    int n = 0;
    std::cout << "Enter number of rows: ";
    std::cin >> n;

    if(n > 10) std::cout << "Value too large!" << std::endl;
    else print_pascal_triangle(n);

    return 0;
}
