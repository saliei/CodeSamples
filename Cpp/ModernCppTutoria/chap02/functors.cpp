#include <iostream>
#include <vector>
#include <assert.h>

// a functor is class that defines the operator()
// this let's create objects that look like a function

struct add_x
{
    add_x(int val): x(val){}
    int operator()(int y) const { return x+y; }

    private:
    int x;
};

int main(int argc, char **argv)
{

    add_x add42(42);
    int i = add42(8);
    assert( i == 50 );

    std::vector<int> in{1, 2, 3};
    std::vector<int> ou(in.size());
    
    std::transform(in.begin(), in.end(), ou.begin(), add_x(1));
    
    for(size_t i = 0; i < ou.size(); i++)
        //assert(ou[i] == in[i]);
        std::cout << in[i] << ", " << ou[i] << std::endl;

    return 0;

}
