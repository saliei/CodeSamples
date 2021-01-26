#include <iostream>
#include <algorithm>
#include <functional>
#include <iterator>
#include <utility>
#include <vector>
#include <random>
#include <chrono>

#include "bst.hpp"

template<typename T, typename U>
std::ostream& operator<<(std::ostream& os, const std::pair<T, U>& p)
{
    os << "(" << p.first << ", " << p.second << ")";
    return os;
}

std::vector<std::pair<int, double>> getRandPairs(long int num=100)
{
    int key, klb=0, kub=100;
    double val, vlb=0, vub=100;
    std::pair<int, double> p;
    std::vector<std::pair<int, double>> vec(num);
    
    std::random_device rd;
    std::default_random_engine eng{rd()};
    std::uniform_int_distribution<int> iunif(klb, kub);
    std::uniform_real_distribution<double> dunif(vlb, vub);
    auto idist = std::bind(iunif, eng);
    auto ddist = std::bind(dunif, eng);
    
    for(int i=0; i<num; ++i)
    {
        key = idist();
        val = ddist();
        p = std::make_pair(key, val);
        vec[i] = p;
    }    

    return vec;
}

void printLine(int num=30)
{
    for(int i=0; i<num; ++i)
        std::cout << "*";
    std::cout << "\n";
}

int main()
{
    Bst<int, double> tree;
    auto vec = getRandPairs(1000000);

    printLine();
    auto t1 = std::chrono::high_resolution_clock::now();
    for(auto p: vec) tree.insert(p);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto et = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    std::cout << "*time to insert " << vec.size() << " pairs: " << et << "ms" << std::endl;
    
    printLine();
    std::cout << "*test insertion by lvalue and rvalue:" << std::endl;
    tree.insert(std::make_pair(1, 1.0));
    auto p1 = std::make_pair(2, 2.0);
    auto it1 = tree.insert(std::move(p1));
    std::cout << "inserted pair: " << (*it1) << std::endl;

    printLine();
    std::cout << "*test post/pre increment/decrement operators:" << std::endl;
    std::cout << "      note: <it> is iterator pointing to inserted pair." << std::endl;
    --it1;
    std::cout << "--it: " << (*it1) << std::endl;
    ++it1;
    std::cout << "++it: " << (*it1) << std::endl;
    it1++;
    std::cout << "it++: " << (*it1) << std::endl;
    it1--;
    std::cout << "it--: " << (*it1) << std::endl;

    printLine();
    std::cout << "*test find methods: " << std::endl;
    auto p2 = vec.back();
    auto it2 = tree.find(p2);
    std::cout << (*it2) << std::endl;
    std::cout << "max pair: " << tree.findMax() << std::endl;
    std::cout << "min pair: " << tree.findMin() << std::endl;

    printLine();
    std::cout << "*test begin and end iterators:" << std::endl;
    auto bg = tree.begin();
    auto cbg = tree.cbegin();
    std::cout << "begin: " << (*bg) << std::endl;
    std::cout << "cbegin: " << (*cbg) << std::endl;
    auto ed = tree.end();
    auto ced = tree.cend();
    --ed;
    std::cout << "--end: " << (*ed) << std::endl;
    std::cout << "vec.back: " << vec.back() << std::endl;



}
