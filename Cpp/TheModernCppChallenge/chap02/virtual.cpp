#include <iostream>

// virtual let's derived class method to overwrite the basic class method.

class Animal
{
    public:
        //void eat() { std::cout << "eats generic food." << std::endl; }
        virtual void eat() { std::cout << "eats generic food." << std::endl; }
};

class Cat: public Animal
{
    public:
        void eat() { std::cout << "eats rat." << std::endl; }
};

void printEat(Animal *xyz){ xyz->eat(); }

int main(int argc, char **argv)
{
    Animal *animal = new Animal;
    Cat *cat = new Cat;

    animal->eat(); // eats generic food.
    cat->eat();    // eats rat.

    printEat(animal); // eats generic food.
    printEat(cat);    // eats generic food. there is an implicit conversion!
}
