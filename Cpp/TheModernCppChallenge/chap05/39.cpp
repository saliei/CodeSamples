#include <iostream>
#include <chrono>
#include <thread>
#include <functional>

template<typename Time = std::chrono::milliseconds, 
         typename Clock = std::chrono::high_resolution_clock>
struct perf_timer
{
    template<typename F, typename... Args>
    static Time duration(F&& f, Args... args)
    {
        auto start = Clock::now();
        std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
        auto end = Clock::now();

        return std::chrono::duration_cast<Time>(end - start);
    }

};

using namespace std::literals::chrono_literals;
void f()
{
    std::this_thread::sleep_for(2s);
}

void g(const int a, const int b)
{
    std::this_thread::sleep_for(1s);
}

int main(int argc, char **argv)
{

    auto t1 = perf_timer<std::chrono::microseconds>::duration(f);
    auto t2 = perf_timer<std::chrono::milliseconds>::duration(g, 2, 3);

    std::cout << "f: " << std::chrono::duration<double, std::micro>(t1).count() << " us." << std::endl;
    std::cout << "g: " << std::chrono::duration<double, std::milli>(t2).count() << " ms." << std::endl;

    auto tot = std::chrono::duration<double, std::nano>(t1 + t2).count();
    std::cout << "tot: " << tot << " ns." << std::endl;

    return 0;

}
