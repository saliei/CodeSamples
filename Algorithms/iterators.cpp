#include <iostream>
#include <vector>
#include <iterator>
#include <string>

using namespace std;

void printVec(const std::vector<std::string>& vec)
{
    // begin will return an iterator or a const_iterator depending on the vector
    // if its a const or not
    // cbegin will return a const_iterator unconditionally

    /*
     *for(std::vector<std::string>::const_iterator iter=vec.cbegin(); iter!=vec.cend();++iter)
     *{
     *    std::cout << *iter << endl;
     *}
     */
    for(auto it=vec.begin(); it!=vec.end(); ++it)
        std::cout << *it << std::endl;
    std::cout << "*****" << std::endl;
    for(auto it=vec.cbegin(); it!=vec.cend(); ++it)
        std::cout << *it << std::endl;
}

int main()
{
    vector<int> nums = {1, 4, 5, 6, 7, 8};
    vector<int>::iterator itr1;
    for(itr1=nums.begin(); itr1!=nums.end(); ++itr1) cout << *itr1 << endl;
    
    cout << "******" << endl;

    vector<int>::iterator itr2 = nums.begin();
    advance(itr2, 2);
    cout << *itr2 << endl;

    cout << "******" << endl;

    auto itr3 = next(itr2, 1);
    cout << *itr3 << endl;
    
    cout << "******" << endl;

    auto itr4 = prev(itr3, 1);
    cout << *itr4 << endl;

    cout << "******" << endl;
    cout << "******" << endl;

    vector<int> nums2 = {2, 3};
    auto itr5 = nums.begin();
    itr5++;
    copy(nums2.begin(), nums2.end(), inserter(nums, itr5));
    for(auto &x: nums) cout << x << endl;
    
    cout << "******" << endl;

    const std::vector<std::string> strs = {"AB", "CD", "PQ", "XY"};
    printVec(strs);

    return 0;
}
