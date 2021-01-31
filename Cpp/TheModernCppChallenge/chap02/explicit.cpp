#include <iostream>
#include <string>

class Entity
{
    public:
        std::string mName;
        int mAge;

        explicit Entity(const std::string &name): mName(name), mAge(-1) {}
        Entity(int age): mName("Unknown"), mAge(age) {}
        //explicit Entity(int age): mName("Unknown"), mAge(age) {}
};

void printEntity(const Entity &entity)
{
    std::cout << "Name: " << entity.mName << ", Age: " << entity.mAge << std::endl;
}

int main()
{
    //Entity a = "Saeid"; //illegal, only one implicit conversion is allowed. 
                          // this is const char *
    //Entity a = (std::string)"Saeid"; // would not compile, needs explicit conversion
    Entity a = (Entity)"Saeid";
    Entity b = 28; // if explicit would not compile

    //printEntity(22); // implicit conversion of 22 to Entity
    //printEntity("Saeid"); //illegal, more than 1 conversion
    
    printEntity(a);

    return 0;

}
