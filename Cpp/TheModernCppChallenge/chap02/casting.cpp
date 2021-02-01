#include <iostream>

class Base
{
    public:
        Base() {}
        virtual ~Base() {}
};

class Derived: public Base
{
    public:
         Derived() {}
        ~Derived() {}
};

class AnotherClass: public Base
{
    public:
        AnotherClass() {}
       ~AnotherClass() {}
};

int main(int argc, char **argv)
{

    int a = 10;
    double b = 5.5;
    int c = b; // c = 5; implicit conversion
    double d = (double)(b + a); // d = 15.5; C-style implicit conversion

    double e = static_cast<int>(b) + a; // C++-style casting


    std::cout << "e: " << e << std::endl;



    std::cin.get();





}
