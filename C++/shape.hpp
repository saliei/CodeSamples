#ifndef SHAPE_H
#define SHAPE_H

class Shape
{   
    /*subclasses inherit these methodes and attributes,
     * you can use these also inside the class def itself,
     * but instances of this class dont have access to this attrs.*/
    protected:
        double height, width;
    public:
        static int numOfShapes;
        Shape(double length);
        Shape(double height, double width);
        Shape();
        /* virtual desctructor ensures that, when deleting
         * any derived class object using a pointer to the base class,
         * it desctruted properly.
         * https://www.geeksforgeeks.org/virtual-destructor/
         */
        virtual ~Shape();
        // setter and getters
        void SetHeight(double height);
        double GetHeight();
        void SetWidth(double width);
        double GetWidth();
        //only static methods can get static attrs
        static int GetNumOfShapes();
        // area func is virtual cause we want to have diff
        // area func for every derived objects from base class Shape
        virtual double Area();

};

#endif
