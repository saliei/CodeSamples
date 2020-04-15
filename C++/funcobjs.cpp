#include <iostream>
#include <functional>
#include <vector>

using namespace std;

double multBy2(double num)
{
    return num*2;
}

double multBy3(double num)
{
    return num*3;
}

// first double is the return type of func 
// second one is the parameter type of func
double doMath(function<double(double)> func, double num)
{
    return func(num);
}

int main()
{
    auto mb2 = multBy2;
    cout << "5*2 = " << mb2(5) << endl;
    cout << "5*2 = " << doMath(multBy2, 5) << endl;
    cout << "5*3 = " << doMath(multBy3, 5) << endl;

    vector<function<double(double)>> funcs(2);
    funcs[0] = multBy2;
    funcs[1] = multBy3;
    cout << "5*2 = " << funcs[0](5) << endl;
    cout << "5*3 = " << funcs[1](5) << endl;
}
