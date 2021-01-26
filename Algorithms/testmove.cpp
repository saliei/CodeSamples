#include <iostream>

class Holder
{
    public:
        Holder(int size): m_size{size}, m_data{new int[size]} {}
        ~Holder() { delete[] m_data; }
        //copy constructor:
        //Holder h1(100);
        //Holder h2 = h1;
        Holder(const Holder& other)
        {
            m_data = new int[other.m_size];
            m_size = other.m_size;
            std::copy(other.m_data, other.m_size+other.m_size, m_data);
        }
        //copy assignment: you want to be able to do this
        // Holder h1(100); ->regular constructor
        // Holder h2(200);
        // h1 = h2; ->h1 should have size of h2 and its data
        // copy constructor should return a refrence to the object
        // you can return by value as opposed to refrence, but this way 
        // compiler makes temp objects to store the temp vals......
        Holder& operator=(const Holder& other)
        {
            if(this == &other) return *this;
            m_size = other.m_size;
            delete[] m_data;
            m_data = new int[other.m_size];
            std::copy(other.m_data, other.m_size+other.m_size, m_data);
            return *this;
        }
        // move semantics takes those temporary created objects
        // the ones returned by value...
        //move constructor
        // imagine createHolder(100) returns Holder(100) by value
        // you want to optimize this operation:
        // Holder h1 = createHolder(100);
        // so that there is no other usage of copy constructor
        Holder(Holder&& other)
        {
            m_size = other.m_size;
            m_data = other.m_data;
            other.m_size = 0;
            other.m_data = nullptr;
        }

        //move assignment
        //Holder h2(100);
        // h2 = creatHolder(200);
        Holder& operator=(Holder&& other)
        {
            if(this == &other) return *this;
            delete[] m_data;
            m_size = other.m_size;
            m_data = other.m_data;
            m_data = nullptr;
            m_size = 0;
            return *this;
        }

    private:
        size_t m_size;
        int* m_data;
}
