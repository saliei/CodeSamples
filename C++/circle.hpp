#ifndef CIRCLE_H
#define CIRCLE_H

#include "shape.hpp"

class Circle: public Shape
{
    public:
        Circle();
        Circle(double width);
        virtual ~Circle();
        double Area();
};

#endif
