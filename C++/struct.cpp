#include <iostream>
#include <cmath>

using namespace std;

// by default all attrs and methods are public in struct
// and private in class defs
struct Shape
{
    double length, width;
    Shape(double l=2, double w=1)
    {
        length = l;
        width = w;
    }

    double Area()
    {
        return length * width;
    }

    private:
        int id;
};

struct Circle: Shape
{
    Circle(double width)
    {
        this->width = width;
    }

    double Area()
    {
        return 3.1415 * pow((width/2), 2);
    }
};

int main()
{
    Shape sh1(4, 5);
    //Shape sh2();
    Circle c1(20);
    Shape rect{10, 20};

    cout << "sh1 Area: " << sh1.Area() << endl;
    //cout << "sh2 Area: " << sh2.Area() << endl;
    cout << "c1 Area:  " << c1.Area()  << endl;
    cout << "rect Area: " << rect.Area() << endl;

    return 0;
}
