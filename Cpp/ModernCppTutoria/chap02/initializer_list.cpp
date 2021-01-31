#include <iostream>
#include <initializer_list>
#include <vector>

// before C++11

class Foo
{
    public:
        int value_a;
        int value_b;
        int value_c;
        Foo(int a, int b, int c): value_a(a), value_b(b), value_c(c) {}
};

class MagicFoo
{
    public:
        std::vector<int> vec;
        MagicFoo(std::initializer_list<int> list)
        {
            for(std::initializer_list<int>::iterator itr = list.begin(); itr != list.end(); itr++)
                vec.push_back(*itr);
        }

        void foo(std::initializer_list<int> list)
        {
            for(std::initializer_list<int>::iterator itr = list.begin(); itr != list.end(); itr++)
                vec.push_back(*itr);
        }
};

int main(int argc, char **argv)
{
    int arr[3] = {1, 2, 3};
    Foo foo(1, 2, 3);
    std::vector<int> vec = {1, 2, 3};

    std::cout << "arr[0]: " << arr[0] << std::endl;
    std::cout << "foo: " << foo.value_a << ", " << foo.value_b << ", " << foo.value_c << std::endl;
    std::cout << "vec: ";
    for(std::vector<int>::iterator itr = vec.begin(); itr != vec.end(); ++itr)
        std::cout << *itr << ", ";
    std::cout << std::endl;

    MagicFoo mfoo  = {1, 2, 3};
    MagicFoo mfoo2 {7, 8, 9}; 
    mfoo.foo({4, 5, 6});
    std::cout << "magic foo: ";
    for(std::vector<int>::iterator itr = mfoo.vec.begin(); itr != mfoo.vec.end(); ++itr)
        std::cout << *itr << ", ";
    std::cout << std::endl;

    std::cout << "magic foo2: ";
    for(std::vector<int>::iterator itr = mfooi2.vec.begin(); itr != mfoo2.vec.end(); ++itr)
        std::cout << *itr << ", ";
    std::cout << std::endl;

    return 0;
}
