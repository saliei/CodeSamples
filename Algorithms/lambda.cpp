#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

void printVecRan(vector<int> v)
{
    for (auto i: v) cout << i << " ";
    cout << endl;
}

void printVecIt(vector<int> v)
{
    for(auto it=v.begin(); it!=v.end(); ++it)
    {
        cout << *it << " ";
    }
    cout << endl;
}

void printVecLam(vector<int> v)
{
    for_each(v.begin(), v.end(), [](int i){
            cout << i << " ";
            });
    cout << endl;
}

vector<int> genRandVec(int nelem, int min, int max)
{
    vector<int> vecVals;
    srand(time(NULL));
    for(int i=0; i < nelem; ++i)
    {
        int val = min + rand() % ((max+1)-min);
        vecVals.push_back(val);
    }
    return vecVals;
}

int main(int argc, char** argv)
{
    vector<int> v {1, 3, 2, 8, 5, 4, 3, 3, 6};
    printVecLam(v);
    printVecIt(v);
    printVecRan(v);
    
    // find first number greater than 4
    // find_if searches for elements which func
    // returns true
    vector<int>::iterator p = find_if(v.begin(), v.end(), [](int i){
            return i > 4;
            });
    cout << "first number greater than 4: " << *p <<endl;

    // compiler can deduce the return type as bool
    // we set here explicitly just for explanation
    sort(v.begin(), v.end(), [](const int& a, const int& b) -> bool{
            // for descending sort use a > b
            return a < b;
            });
    printVecLam(v);

    // count num of elems greater than 5
    int count = count_if(v.begin(), v.end(), [](int i){
            return i > 5;
            });
    cout << "num of elems greater than 5: " << count << endl;
    
    vector<int> rv = genRandVec(10, 4, 20);
    printVecLam(rv);
    //TODO: apparently there is no diff between const int& a, ?
    sort(rv.begin(), rv.end(), [](int a, int b){
            return a < b;
            });
    printVecRan(rv);

    //sum of all elems, [&]: capture all vars by ref
    int sum=0;
    for_each(rv.begin(), rv.end(), [&](int x){sum += x;});
    cout << "sum of vec: " << sum << endl;
    
    return 0;   
}
