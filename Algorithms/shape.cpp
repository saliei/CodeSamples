#include "shape.hpp"

Shape::Shape(double length)
{
    this->height = length;
    this->width = length;
}

Shape::Shape(double height, double width)
{
    this->height = height;
    this->width = width;
}

Shape::~Shape() = default;

void Shape::SetHeight(double height)
{
    this->height = height;
}

double Shape::GetHeight() {return height;}

void Shape::SetWidth(double width)
{
    this->width = width;
}

double Shape::GetWidth() {return width;}

int Shape::GetNumOfShapes() {return numOfShapes;}

//polymorphisim, any virtual method can be used 'polymorphically'
double Shape::Area()
{
    return height * width;
}

// default num of shapes
int Shape::numOfShapes = 0;
