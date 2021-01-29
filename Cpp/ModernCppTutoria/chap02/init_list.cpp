#include <iostream>
#include <vector>
#include <initializer_list>

template<typename T>
class MagicFoo
{
    public:
       std::vector<T> vec;
       MagicFoo(std::initializer_list<T> list)
       {
           for(typename std::initializer_list<T>::iterator iter = list.begin(); iter != list.end(); iter++)
               vec.push_back(*iter);
       } 
};

int main()
{
    MagicFoo<int> magicfoo = {1, 2, 4, 8};
    std::cout << "magicfoo: ";
    for(std::vector<int>::iterator iter = magicfoo.vec.begin(); iter != magicfoo.vec.end(); iter++)
        std::cout << *iter << ", ";
    std::cout << std::endl;
}
