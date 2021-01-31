#include <iostream>
#include <vector>

void func1(std::vector<double> &v)
{
    std::transform(v.begin(), v.end(), v.begin(), [](double d) { return d < 0.001 ? 0: d; });
}

void func2(std::vector<double> &v)
{
    std::transform(std::begin(v), std::end(v), std::begin(v), [](double d) -> double { 
            if(d < 0.001) return 0;
            else return d; }  );
}

void func3(std::vector<double> &v, const double &epsilon)
{
    std::transform(v.begin(), v.end(), v.begin(), [&, epsilon](double d) mutable -> double
            {
                if( d < epsilon ) return 0;
                else return d;
            });
}

void print_vec(std::vector<double> &v)
{
    for(std::vector<double>::iterator it = v.begin(); it != v.end(); ++it)
        std::cout << *it << " ";
    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    std::vector<double> vec{1.001, 0.0001, 0.001, 2.0};
    print_vec(vec);
    
    func1(vec);
    print_vec(vec);
    
    func2(vec);
    print_vec(vec);
    
    func3(vec, 0.001);
    print_vec(vec);

    return 0;

}
