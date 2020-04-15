#include <iostream>

using namespace std;

int global=10;

int setValue()
{
    return 5;
}

/*
 *int setGlobal()
 *{
 *    return global;
 *}
 */

int& setGlobal()
{
    return global;
}

int fun(int& x)
{
    cout << "fun(int& x)" << endl;
    return 0;
}

int fun(const int& x)
{
    cout << "fun(const int& x)" << endl;
    return 0;
}

int main()
{
    //setValue() = 20; //ERROR!
    setGlobal() = 20;
    cout << "global: " << global << endl;
    
    //lvalue to rvalue conversion
    int x = 1;
    int y = 2;
    //+ operator takes two rvalues => x and y get 
    //implicitly converted to rvalues
    int z = x + y;

    int q = 8;
    int& qref = q; //qref is of type int&, a refrence to an integer
    cout << "qref: " << qref << endl;
    qref++; // qref is now 9
    cout << "qref: " << qref << endl;
    //int& qref = 8; // ERROR! 8 is no where its an rvalue in some register, i.e. no conversion from rvalue to lvalue
    //fun(10); // ERROR! will fail for the same reason, no rvalue to lvalue conversion(fun takes an lvalue ref to an int)
    int p = 10; //this works!
    fun(p);

    // you are allowed to bind const lvalues to an rvalue
    const int& ref = 50;
    cout << "ref: " << ref << endl;;
    //ref++; //ERROR! increment of read-only refrence 
    fun(10);

    const int& r = 100; //compiler creats an aux var
    //int ___internal = r;
    //const int& r = ___internal;



}
