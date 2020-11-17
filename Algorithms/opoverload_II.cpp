#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>

using namespace std;

class Box
{
    public:
        double length, width, breadth;
        string boxString;
        Box(){length=1, width=1, breadth=1;}
        Box(double l, double w, double b) {length=l, width=w, breadth=b;}

        Box& operator++()
        {
            length++;
            width++;
            breadth++;
            return *this;
        }

        operator const char*()
        {
            ostringstream boxStream;
            boxStream << "Box: " << length << ", " <<
                width << ", " << breadth;
            boxString = boxStream.str();
            return boxString.c_str();
        }

        Box operator+(const Box& othBox)
        {
            Box resBox;
            resBox.length = length + othBox.length;
            resBox.width = width + othBox.width;
            resBox.breadth = breadth + othBox.breadth;
            return resBox;
        }

        bool operator==(const Box& othBox)
        {
            return ((length==othBox.length) &&
                    (width==othBox.width) &&
                    (breadth==othBox.breadth));
        }
};

int main(int argc, char** argv)
{
    Box b1(10, 15, 20);
    cout << b1 << endl;
    ++b1;
    cout << b1 << endl;
    Box b2(20, 15, 10);
    cout << b2 << endl;
    Box b3 = b1 + b2;
    cout << b3 << endl;
    cout << "b1 == b2: " << (b1==b2) << endl;

    return 0;
}
