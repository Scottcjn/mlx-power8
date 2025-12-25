// MLX SIMD Type Selection for Linux
// Copyright © 2025 Elyan Labs - Multi-architecture support

#pragma once

#include "mlx/backend/cpu/simd/base_simd.h"

// x86_64 with AVX
#if defined(__x86_64__) && defined(__AVX__)
#include "mlx/backend/cpu/simd/avx_simd.h"
#endif

// POWER8/9 with VSX
#if defined(__powerpc64__) && defined(__VSX__)
#include "mlx/backend/cpu/simd/vsx_simd.h"
#endif

// ARM64 with NEON (already in upstream)
#if defined(__aarch64__)
// Use upstream neon_fp16_simd.h or accelerate_simd.h
#endif
