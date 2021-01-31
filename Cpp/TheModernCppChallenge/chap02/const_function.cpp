#include <iostream>

// `const` specification after a function means that
// this function can not modify member data, unless it's `mutable`.
// `this` pointer will essentially becomes a pointer to `const` object
// the `const` is part of the function signature, which means you can have 
// two similar methods, one which is called when the object is `const` and
// one that isn't.


class Entity
{
    private:
        int counter1;
        mutable int counter2;
    public:
        
        Entity(): counter1(0), counter2(0) {}

        void Foo() 
        { 
            counter1++; //this works
            counter2++;
            std::cout << "Foo" << std::endl;
        }
        void Foo() const
        { 
            //counter1++; // increment of read-only object
            counter2++;
            std::cout << "Foo const" << std::endl; 
        }

        int getInvocations() const { return counter2; }
};

int main(int argc, char **argv)
{
    Entity en;
    const Entity &cen = en;

     en.Foo();
    cen.Foo();
    std::cout << "Foo has been called: " << cen.getInvocations() << " times" << std::endl;
}
