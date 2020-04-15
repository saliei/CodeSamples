#include <iostream>
#include <stdio.h>
#include <memory>

using namespace std;

class Entity
{
    public:
        Entity() { cout << "Entity created." << endl; }
       void  Print() { cout << "Print func." << endl; }
        ~Entity() { cout << "Entity destroyed." << endl; }

};

template <typename T>
class SmartPtr
{
    private:
        T *ptr;
    public:
        explicit SmartPtr(T *p = NULL): ptr(p) {}
        ~SmartPtr() { delete(ptr); }
        //overload derefrence operator
        T& operator*() { return *ptr; }
        T* operator->() { return ptr; }
};

int main()
{
    int numElem = 10;
    int* arr = (int*)malloc(numElem * sizeof(int));
    for(int i=0; i < numElem; ++i) arr[i] = i;
    for(int i=0; i < numElem; ++i) cout << arr[i] << endl;
    free(arr);

    cout << "************" << endl;

    // no need to delete ptr, check with valgrind
    // ptr(new int) also works
    SmartPtr<int> ptr(new int());
    *ptr = 20;
    cout << *ptr << endl;
   
    cout << "************" << endl;
    
    // you can't copy unique_ptr, = operator
    // is overloaded to delete
    unique_ptr<int[]> nums(new int[numElem]);
    if(nums != NULL)
    {
        for(int i=0; i < numElem; ++i)
            nums[i] = i;
    }
    for(int i=0; i < numElem; ++i) cout << nums[i] << endl;

    cout << "************" << endl;
    
    // a new scope
    std::shared_ptr<Entity> shaE;
    {
        // next def won't work, cause unique_ptr constructor is explicit
        //std::unique_ptr<Entity> entity = new Entity();
        
        // this is fine, but prefered way is the next one
        //std::unique_ptr<Entity> entity(new Entity());
        
        // preferd way due to exception safety
        std::unique_ptr<Entity> entity = std::make_unique<Entity>();
        entity->Print();

        //shared_ptr can be copied, it uses refrence counters to track
        //all the copies, imagine you have two copies of the pointer,
        //so ref counter is two, if one goes out of scope ref counter
        //becomes 1, and when the other goes out of scope too, the pointer
        //gets deltes.
        //also sharedEntity(new Entity()) works, but strongly prohibited
        //refer to: 
        //https://www.youtube.com/watch?v=UOB7-B2MfwA
        std::shared_ptr<Entity> sharedEntity = std::make_shared<Entity>();
        shaE = sharedEntity;
        
        //this doesn't increase the ref counter, as opposed to shared_ptr
        //when you assign a shared_ptr to a weak_ptr is wont increase the
        //ref counter
        std::weak_ptr<Entity> weakEntity = sharedEntity;

    }

    return 0;
}

