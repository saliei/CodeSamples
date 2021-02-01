#include <iostream>

// a `friend` function can access private and protected data of a class.

class EntityB;

class EntityA
{
    private:
        int member;
        friend int addFive(EntityA);
        friend int addTwoClasses(EntityA, EntityB);
    public:
        EntityA (const int num): member(num) {}
        ~EntityA() {}
};

class EntityB
{
    private:
        int member;
        friend int addTwoClasses(EntityA, EntityB);
    public:
        EntityB(const int num): member(num) {}
        ~EntityB(){}
};

int addFive(EntityA e)
{
    e.member += 5;
    return e.member;
}

int addTwoClasses(EntityA a, EntityB b)
{
    return a.member + b.member;
}

int main(int argc, char **argv)
{

    EntityA obja(10);
    int a = addFive(obja);
   // std::cout << obja.member << std::endl; // error: ‘int EntityA::member’ is private within this context
    std::cout << a << std::endl;

    EntityB objb(20);
    int res = addTwoClasses(obja, objb);
    std::cout << res << std::endl;

    return 0;
    

}
