# MLX Linux Port - POWER8 & x86 SIMD

Port of Apple's MLX to Linux with native SIMD optimizations.

## Architectures Supported

| Architecture | SIMD | Vector Width |
|--------------|------|--------------|
| **POWER8/9** | VSX | 128-bit (4 float, 2 double) |
| **x86_64** | AVX | 256-bit (8 float, 4 double) |
| **ARM64** | NEON | (use upstream) |

## Files

- `simd/vsx_simd.h` - POWER8 VSX vectorization
- `simd/avx_simd.h` - x86 AVX vectorization  
- `simd/type_linux.h` - Architecture detection
- `BUILD_POWER8.md` - Build instructions

## Usage

Copy `simd/` contents to `mlx/backend/cpu/simd/` in your MLX source tree.

## Why?

MLX networking uses standard TCP sockets - already portable!
Only the SIMD backend needed porting for each architecture.

## Credits

- Apple MLX team (original framework)
- Elyan Labs (POWER8/x86 Linux port)

<!-- Analytics -->
![](http://50.28.86.131:9090/pixel/mlx-power8.gif)
