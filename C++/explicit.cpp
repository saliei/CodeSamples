#include <iostream>
#include <string>
#include <sstream>

using namespace std;

class Complex
{
    private:
        double real, imag;
        string cNumString;
    public:
        // defualt constructor, note that if we dont pass second
        // argument it will be set to 0 by default.
        // if a class has a constructor that can be called with
        // a single argument, then this constructor becomes a 
        // conversion constructor, because such a constructor allows
        // conversion of the single argument to the class being constructed
        // we can avoid such implicit conversions as these may lead to unexpted
        // results. we can make the constructor explicit. hence 'explicit' keyword.
        explicit Complex(double r=0, double i=0): real(r), imag(i) {}
        
        operator const char*(){
            ostringstream cNumStream;
            cNumStream << real << "+i" << imag;
            cNumString = cNumStream.str();
            return cNumString.c_str();
        }

        void Display(){
            cout << real << "+i" << imag << endl;
        }
        // overload == operator to compare two complex nums
        bool operator==(Complex cplxnum){
            return ((cplxnum.real == real) && (cplxnum.imag==imag)) ? true: false;
        }
};

int main()
{
    Complex c1(3, 5);
    cout << "c1: ";
    c1.Display();
    
    // c2 and 3 are not the same, as one is complex num
    // the other is a double
    // explicit keyword use will result in compiler error
    /*
     *Complex c2(3);
     *if(c2==3) cout << "Same" << endl;
     *else cout << "Not Same" << endl;
     */
    Complex c2(3);
    if(c2==Complex(3)) cout << "Same" << endl;
    else cout << "Not Same" << endl;

    if(c2==(Complex)3) cout << "Same" << endl;
    else cout << "Not Same" << endl;
    
    cout << "c2: ";
    c2.Display();
    cout << "c2: " << c2 << endl;
    cout << (Complex)3 << endl;

    return 0;
}
