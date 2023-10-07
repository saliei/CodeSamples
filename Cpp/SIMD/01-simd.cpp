#include <iostream>
#include <vector>
#include <immintrin.h> // AVX intrinsic functions

// x86 simd instructions:
// MMX / SSE / AVX / AVX2 (256bit registers) / AVX512 (512bit registers)

using vec = std::vector<float>;

vec simd_add(const vec& a, const vec& b) {
  vec c(a.size());

  constexpr auto FLOATS_IN_AVX_REGISTER = 8u; // 8 floats in a 256-bit avx register 
  const auto vectorizable_nelems = (a.size() / FLOATS_IN_AVX_REGISTER) * FLOATS_IN_AVX_REGISTER;

  auto i = 0u;
  for(; i < vectorizable_nelems; i += FLOATS_IN_AVX_REGISTER) {
    // load those floats into avx registers
    auto a_reg = _mm256_loadu_ps(a.data() + i);
    auto b_reg = _mm256_loadu_ps(b.data() + i);
  
    auto intermediate_sum = _mm256_add_ps(a_reg, b_reg);

    _mm256_storeu_ps(c.data() + i, intermediate_sum);
  }

  // handle the remaining elements that wasn't handled in vectorized fashion
  for(; i < c.size(); ++i) {
    c[i] = a[i] + b[i];
  }

  return c;
}

int main (int argc, char *argv[]) {
  
  vec a(19, 1.0f);
  auto res = simd_add(a, a);
  std::cout << "count of result: " << res.size() << std::endl;
  for(const auto elm: res) {
    std::cout << elm << ", ";
  }
  std::cout << std::endl;

  return 0;
}
