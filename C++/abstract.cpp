#include <iostream>

using namespace std;

// an abstract class, sole purpose is to be a base class
// for others, a class is an abstract one if it has at least
// one pure virtual function.
class Base
{
    protected:
        int x;
    public:
        //pure vitual function
        virtual void printVars() = 0;
        Base(int i) {x = i;}
};

class Derived: public Base
{
    private:
        int y;
    public:
        Derived(int i, int j): Base(i) {y = j;}
        void printVars() override 
        {
            cout << "x = " << x << ", y= " << y << endl;
        }
};

int main()
{
    Derived d(4, 5);
    d.printVars();
    Base *bp = new Derived(9, 8);
    bp->printVars();

    return 0;
}
