#include <iostream>
#include <vector>
#include <assert.h>


template<class T>
class circular_buffer
{
    typedef circular_buffer_iterator<T> const_iterator;

    circular_buffer() = delete;

    
  public:
    explicit circular_buffer(size_t const size): data_(size) {}
    bool clear() noexcept { head_ = -1; size_ = 0; }
    bool empty() const noexcept { return size_ = 0; }









};

int main(int argc, char **argv)
{

}
