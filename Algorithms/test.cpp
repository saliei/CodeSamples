#include <iostream>

using namespace std;

template <typename Key, typename Val>
struct BstNode
{
    using Pair = std::pair<Key, Val>;
        
    Pair data;
    BstNode* right;
    BstNode* left;
    BstNode* parent;

    BstNode(const Pair& dat, BstNode* rn, BstNode* ln, BstNode* par): 
        data(dat), right(rn), left(ln), parent(par) {}

    BstNode(Pair&& dat, BstNode* rn, BstNode* ln, BstNode* par):
        data(std::move(dat)), right(rn), left(ln), parent(par) {}
};

template <typename Key, typename Val, typename Cmp>
void test(const std::pair<Key, Val>&)
{
    
}

template <typename T, typename U, typename Cmp=std::less<T>>
bool compare(const T& t, const U& u)
{
    /*
     *Cmp c;
     *auto q = c(t, u);
     *cout <<  q << endl;
     */
    Cmp c;
    return c(t, u);
}

int main()
{
    compare(10, 20);
    //compare(5, 10);
    std::pair<string, string> p1 = std::make_pair("10", "R");
    std::pair<string, string> p2 = std::make_pair("5", "E");
    std::pair<int, int> p3 = std::make_pair(10, 30);
    std::pair<int, int> p4 = std::make_pair(10, 20);

    cout << (p1>p2) << endl;
    cout << (p3==p4) << endl;
    /*
     *if(compare(30, 50)) cout << "true" << endl;
     *else if(compare(50, 30)) cout << "false" << endl;
     */
}
