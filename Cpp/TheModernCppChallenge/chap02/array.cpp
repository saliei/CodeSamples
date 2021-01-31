#include <iostream>
#include <array>

// std::array does not decay to a pointer
// has a fixed size, should be passed around


void getSize(const std::array<double, 5> &myArray)
{
    std::cout << "size: " << myArray.size() << std::endl;
}

int main(int argc, char **argv)
{
    std::array myArray {1.0, 2.0, 3.3, 4.0, 5.0}; // list initialization
    std::array<int, 2> myArray2 = {1, 2}; // initializer list
    getSize(myArray);

    return 0;

}
