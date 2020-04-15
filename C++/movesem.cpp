#include <iostream>

std::string getString(){
    return "Hello World!";
}

class Holder
{
    private:
        int* m_data;
        size_t m_size;
    public:
        Holder(int size)
        {
            m_data = new int[size];
            m_size = size;
        }

        ~Holder() { delete[] m_data; }

        // copy constructor
        Holder(const Holder& other)
        {
            m_data = new int[other.m_size];
            std::copy(other.m_data, other.m_data+other.m_size, m_data);
            m_size = other.m_size;
        }

        // assignment operator
        // by convension a ref to this class is returned
        // the key point of copy and assignment operator is that
        // they both receive const ref to an object in input and 
        // make a copy out of it for the class they belong to
        Holder& operator=(const Holder& other)
        {
            if(this == &other) return *this;
            delete[] m_data;
            m_data = new int[other.m_size];
            std::copy(other.m_data, other.m_data+other.m_size, m_data);
            m_size = other.m_size;
            return *this;
        }

        // move constructor
        // it takes an rvalue ref to another Holder object,
        // being an rvalue ref we can modify it. so we steal it's
        // data, then set it to null, no deep copies here we just moved
        // resources around
        Holder(Holder&& other)
        {
            m_data = other.m_data;
            m_size = other.m_size;
            other.m_data = nullptr;
            other.m_size = 0;
        }

        // move assignment operator
        Holder& operator=(Holder&& other)
        {
            if(this == &other) return *this;
            delete[] m_data;
            m_data = other.m_data;
            m_size = other.m_size;
            other.m_data = nullptr;
            other.m_size = 0;
            return *this;
        }

        void getSize() { std::cout << "size: " << m_size << std::endl; }
};

// this returns a Holder object by value,
// when a function returns an object by value,
// the compiler has to create a temporary,
// yet fully-fledged object(rvalue). Holder
// object is a heavy-weight object due to its internal
// memory allocation, which is a very expensive task
Holder createHolder(int size)
{
    return Holder(size);
}

int main()
{
    // 666: an rvalue, non mem addr, temp place on
    // some registry while prog running. to be able
    // to use its mem addr it must be stores in an
    // lavlue, which is x
    int x = 666;
    // compiler stores result of + operator in rhs
    // in some temp object, output is an rvalue.
    // hence storing in lvalue y
    int y = x + 10;

    std::string s1 = "Hello ";
    std::string s2 = "World!";
    std::string s3 = s1 + s2;
    // rhs is an rvalue as it returns from
    // function, its rvalue but not directly
    std::string s4 = getString();
    
    // Error:
    // you are allowed to take the addr
    // of an rvalue only if you store it in a const
    //int& x = 5;
    const int& a = 5;
    std::cout << a << std::endl;
    
    // rvalue ref, its like removing const above
    // lvalue ref is like traditional refrence
    // rvalue ref is a new type introduced in c++11
    // str_rref is a ref to temp object, no const
    // do we can modify it
    std::string&& str_rref = s1 + s2;
    str_rref += " , Freind!";
    std::cout << str_rref << std::endl;
    
    Holder h1(10);  // regular constructor
    Holder h2 = h1; // copy constructor
    Holder h3(h1);  // copy constructor (alternative syntax)
    h1.getSize();
    h2.getSize();
    h3.getSize();

    Holder h4(20);
    h4.getSize();
    h4 = h1; // assignment operator
    h4.getSize();
    
    // a temp object coming out of createHolder is 
    // passed to the copy constructor which itself allocates
    // its own m_data pointer by copying the data from the temp
    // object. hence two expensive mem allocation:
    // a) during the creating of the temp object
    // b) during the actual object copy-constructor operation
    // same goes for the assignment operation.
    // instead we can steal/move allocated data during the 
    // construnction/assignment stages.
    Holder h5 = createHolder(30);
    h5.getSize();

    Holder h6(60); //regular constructor
    Holder h7(h6); //copy constructor, lvalue
    Holder h8 = createHolder(80); //move consrtructor, rvalue
    h6.getSize();
    h7.getSize();
    h8.getSize();
    h7 = h8; //assignment operator
    h7.getSize();
    h7 = createHolder(70);
    h7.getSize();
    
    Holder h10(100); //h10 is an lvalue
    Holder h11(h10); //copy consrtructor invoked because of lavalue input
    Holder h12(std::move(h10)); // move constructor, rvalue input
    h11.getSize();
    h12.getSize();

    return 0;
}
