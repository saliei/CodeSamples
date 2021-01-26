#include "circle.hpp"
#include <cmath>

Circle::Circle(double width): Shape(width){}

Circle::~Circle() = default;

double Circle::Area()
{
    return 3.1415 * pow((width / 2), 2);
}
