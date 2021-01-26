#include <iostream>

using namespace std;

/*
 *template <typename T>
 *T times2(T val)
 *{
 *    return val*2;
 *}
 */

template <typename T>
void times2(T val)
{
    cout << "2*" << val << "= " << val*2 << endl;
}

template <typename T>
T add(T x, T y)
{
    return x+y;
}

template <typename T>
T Max(T x, T y)
{
    return (x < y) ? y: x;
}

template <typename T, typename U>
class Person
{
    public:
        T weight;
        U height;
        static int numOfPeople;
        Person(T w, U h)
        {
            weight = w;
            height = h;
            numOfPeople++;
        }

        void Display()
        {
            cout << "wight= " << weight <<
                " height= " << height << endl;
        }
};

template <typename T, int N>
class Array
{
    private:
        T arr[N];
    public:
        const int GetSize() { return N; }
};

template <typename T, typename U> int Person<T, U>::numOfPeople;

int main()
{
    /*
     *auto q = times2(2);
     *auto p = times2(2.2);
     *cout << q << endl;
     *cout << p << endl;
     */
    times2(3);
    times2(3.1);

    cout << "5 + 2 = " << add(5, 2) << endl;
    cout << "5.1 + 2.1 = " << add(5.1, 2.1) << endl;
    
    cout << "Max 8, 2 = " << Max(8, 2) << endl;
    cout << "Max of 'cat', 'dog' = " << Max("cat", "dog") << endl;
    
    Person<double, int> mk(5.1, 200);
    mk.Display();
    cout << "numOfPeople = " << mk.numOfPeople << endl;

    // an array that's size is known at comile time
    // and is templated
    Array<double, 10> arr;
    cout << "Size of arr with templates: " << arr.GetSize() << endl;

    return 0;
}
