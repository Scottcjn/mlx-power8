// Copyright © 2025 Elyan Labs - POWER8 VSX SIMD for MLX
// MIT License (same as MLX)

#pragma once

#ifdef __powerpc64__

#include <altivec.h>
#include <stdint.h>

namespace mlx::core::simd {

// POWER8 VSX vector types
template <>
struct Simd<float, 4> {
  static constexpr int size = 4;
  __vector float value;
  
  Simd() {}
  Simd(__vector float v) : value(v) {}
  Simd(float v) : value(vec_splats(v)) {}
};

template <>
struct Simd<double, 2> {
  static constexpr int size = 2;
  __vector double value;
  
  Simd() {}
  Simd(__vector double v) : value(v) {}
  Simd(double v) : value(vec_splats(v)) {}
};

template <>
struct Simd<int32_t, 4> {
  static constexpr int size = 4;
  __vector int value;
  
  Simd() {}
  Simd(__vector int v) : value(v) {}
  Simd(int32_t v) : value(vec_splats(v)) {}
};

// Max SIMD sizes for POWER8 VSX
template <> static constexpr int max_size<float> = 4;
template <> static constexpr int max_size<double> = 2;
template <> static constexpr int max_size<int32_t> = 4;

// Load/Store
template <>
Simd<float, 4> load(const float* x) {
  return Simd<float, 4>{vec_vsx_ld(0, x)};
}

template <>
void store(float* dst, Simd<float, 4> x) {
  vec_vsx_st(x.value, 0, dst);
}

template <>
Simd<double, 2> load(const double* x) {
  return Simd<double, 2>{vec_vsx_ld(0, x)};
}

template <>
void store(double* dst, Simd<double, 2> x) {
  vec_vsx_st(x.value, 0, dst);
}

// Arithmetic operations - float4
inline Simd<float, 4> operator+(Simd<float, 4> a, Simd<float, 4> b) {
  return Simd<float, 4>{vec_add(a.value, b.value)};
}

inline Simd<float, 4> operator-(Simd<float, 4> a, Simd<float, 4> b) {
  return Simd<float, 4>{vec_sub(a.value, b.value)};
}

inline Simd<float, 4> operator*(Simd<float, 4> a, Simd<float, 4> b) {
  return Simd<float, 4>{vec_mul(a.value, b.value)};
}

inline Simd<float, 4> operator/(Simd<float, 4> a, Simd<float, 4> b) {
  return Simd<float, 4>{vec_div(a.value, b.value)};
}

// FMA - Fused Multiply-Add (POWER8 excels at this!)
inline Simd<float, 4> fma(Simd<float, 4> a, Simd<float, 4> b, Simd<float, 4> c) {
  return Simd<float, 4>{vec_madd(a.value, b.value, c.value)};
}

// Sqrt
inline Simd<float, 4> sqrt(Simd<float, 4> x) {
  return Simd<float, 4>{vec_sqrt(x.value)};
}

// Rsqrt (1/sqrt) - POWER8 has hardware rsqrt estimate
inline Simd<float, 4> rsqrt(Simd<float, 4> x) {
  return Simd<float, 4>{vec_rsqrte(x.value)};
}

// Max/Min
inline Simd<float, 4> max(Simd<float, 4> a, Simd<float, 4> b) {
  return Simd<float, 4>{vec_max(a.value, b.value)};
}

inline Simd<float, 4> min(Simd<float, 4> a, Simd<float, 4> b) {
  return Simd<float, 4>{vec_min(a.value, b.value)};
}

// Abs
inline Simd<float, 4> abs(Simd<float, 4> x) {
  return Simd<float, 4>{vec_abs(x.value)};
}

} // namespace mlx::core::simd

#endif // __powerpc64__
