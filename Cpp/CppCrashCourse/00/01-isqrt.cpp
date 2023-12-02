#include <cstdio>

constexpr int isqrt(int n) {
    int i = 1;
    while(i*i < n) ++i;
    return i - (i*i != n);
}

int main() {
    // we could have manually inserted the square root of 1764 in the code, but...
    constexpr int s = isqrt(1764);
    printf("%d\n", s);

    return 0;
}
