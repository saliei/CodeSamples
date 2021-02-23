#include <iostream>
#include <functional>
#include <algorithm>
#include <assert.h>
#include <vector>

template<class T, class Compare=std::less<typename std::vector<T>::value_type>>
class priority_queue
{
    typedef typename std::vector<T>::value_type value_type;
    typedef typename std::vector<T>::size_type size_type;
    typedef typename std::vector<T>::reference reference;
    typedef typename std::vector<T>::const_reference const_reference;

    private:
        std::vector<T> data;
        Compare comparer;
    public:
        bool empty() const noexcept { return data.empty(); }
        
        size_type size() const noexcept { return data.size(); }

        void push(const value_type& value)
        {
            data.push_back(value);
            std::push_heap(std::begin(data), std::end(data), comparer);
        }

        void pop()
        {
            std::pop_heap(std::begin(data), std::end(data), comparer);
            data.pop_back();
        }

        const_reference top() const { return data.front(); }

        void swap(priority_queue& other ) noexcept
        {
            swap(data, other.data);
            swap(comparer, other.comparer);
        }
};

template<class T, class Compare>
void swap(priority_queue<T, Compare>& lhs, priority_queue<T, Compare>& rhs) noexcept { lhs.swap(rhs); }

int main(int argc, char **argv)
{
    priority_queue<int> q;

    for(int i: {1, 4, 5, 5, 6, 2, 3})
    {
        q.push(i);
    }

    assert(!q.empty());
    assert(q.size() == 7);

    while(!q.empty())
    {
        std::cout << q.top() << " ";
        q.pop();
    }

    std::cout << std::endl;

    return 0;

}
