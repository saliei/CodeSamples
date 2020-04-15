#include <iostream>

using namespace std;

class Complex
{
    private:
        double real, imag;
    public:
        Complex(double r=0, double i=0)
        {
            real = r;
            imag = i;
        }

        void print()
        {
            cout << real << "+i" << imag << endl;
        }

        // global operator function is made friend of this class 
        // so that it can access its private members
        friend Complex operator+(Complex const&, Complex const&);
};

Complex operator+(Complex const &c1, Complex const &c2)
{
    return Complex(c1.real+c2.real, c1.imag+c2.imag);
}

int main()
{
    Complex c1(1, 2);
    Complex c2(3, 6);
    Complex c3 = c1 + c2;
    c3.print();

    return 0;
}
