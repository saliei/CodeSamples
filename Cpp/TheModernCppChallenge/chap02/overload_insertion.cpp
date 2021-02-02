#include <iostream>

// the overload << operator function must then be declared as
// friend of the class, so it can access it's private data.
// https://docs.microsoft.com/en-us/cpp/standard-library/ 
// overloading-the-output-operator-for-your-own-classes?view=msvc-160


class Date
{
    int month, day, year;
  public:
    Date(int m, int d, int y): month(m), day(d), year(y) {}
    friend std::ostream& operator<<(std::ostream &os, const Date &date);
};

std::ostream& operator<<(std::ostream &os, const Date &date)
{
    os << date.month << "/" << date.day << "/" << date.year << std::endl;
    return os;
}

int main(int argc, char **argv)
{
    Date date(1, 29, 2021);
    std::cout << date ;

    return 0;
}
