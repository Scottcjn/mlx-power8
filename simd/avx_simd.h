// Copyright © 2025 Elyan Labs - x86 AVX SIMD for MLX Linux
// MIT License (same as MLX)

#pragma once

#if defined(__x86_64__) && defined(__AVX__)

#include <immintrin.h>
#include <stdint.h>

namespace mlx::core::simd {

// x86 AVX vector types - 256-bit (8 floats, 4 doubles)
template <>
struct Simd<float, 8> {
  static constexpr int size = 8;
  __m256 value;
  
  Simd() {}
  Simd(__m256 v) : value(v) {}
  Simd(float v) : value(_mm256_set1_ps(v)) {}
};

template <>
struct Simd<double, 4> {
  static constexpr int size = 4;
  __m256d value;
  
  Simd() {}
  Simd(__m256d v) : value(v) {}
  Simd(double v) : value(_mm256_set1_pd(v)) {}
};

// SSE fallback - 128-bit (4 floats, 2 doubles)
template <>
struct Simd<float, 4> {
  static constexpr int size = 4;
  __m128 value;
  
  Simd() {}
  Simd(__m128 v) : value(v) {}
  Simd(float v) : value(_mm_set1_ps(v)) {}
};

// Max SIMD sizes for x86 AVX
template <> static constexpr int max_size<float> = 8;
template <> static constexpr int max_size<double> = 4;

// Load/Store - AVX 256-bit
template <>
Simd<float, 8> load(const float* x) {
  return Simd<float, 8>{_mm256_loadu_ps(x)};
}

template <>
void store(float* dst, Simd<float, 8> x) {
  _mm256_storeu_ps(dst, x.value);
}

template <>
Simd<double, 4> load(const double* x) {
  return Simd<double, 4>{_mm256_loadu_pd(x)};
}

template <>
void store(double* dst, Simd<double, 4> x) {
  _mm256_storeu_pd(dst, x.value);
}

// Arithmetic - float8 AVX
inline Simd<float, 8> operator+(Simd<float, 8> a, Simd<float, 8> b) {
  return Simd<float, 8>{_mm256_add_ps(a.value, b.value)};
}

inline Simd<float, 8> operator-(Simd<float, 8> a, Simd<float, 8> b) {
  return Simd<float, 8>{_mm256_sub_ps(a.value, b.value)};
}

inline Simd<float, 8> operator*(Simd<float, 8> a, Simd<float, 8> b) {
  return Simd<float, 8>{_mm256_mul_ps(a.value, b.value)};
}

inline Simd<float, 8> operator/(Simd<float, 8> a, Simd<float, 8> b) {
  return Simd<float, 8>{_mm256_div_ps(a.value, b.value)};
}

// FMA - Fused Multiply-Add (AVX2/FMA)
#ifdef __FMA__
inline Simd<float, 8> fma(Simd<float, 8> a, Simd<float, 8> b, Simd<float, 8> c) {
  return Simd<float, 8>{_mm256_fmadd_ps(a.value, b.value, c.value)};
}
#endif

// Sqrt
inline Simd<float, 8> sqrt(Simd<float, 8> x) {
  return Simd<float, 8>{_mm256_sqrt_ps(x.value)};
}

// Rsqrt (1/sqrt) - approximate
inline Simd<float, 8> rsqrt(Simd<float, 8> x) {
  return Simd<float, 8>{_mm256_rsqrt_ps(x.value)};
}

// Max/Min
inline Simd<float, 8> max(Simd<float, 8> a, Simd<float, 8> b) {
  return Simd<float, 8>{_mm256_max_ps(a.value, b.value)};
}

inline Simd<float, 8> min(Simd<float, 8> a, Simd<float, 8> b) {
  return Simd<float, 8>{_mm256_min_ps(a.value, b.value)};
}

// Abs (clear sign bit)
inline Simd<float, 8> abs(Simd<float, 8> x) {
  __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
  return Simd<float, 8>{_mm256_and_ps(x.value, mask)};
}

} // namespace mlx::core::simd

#endif // __x86_64__ && __AVX__
