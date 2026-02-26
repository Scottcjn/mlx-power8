[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![MLX](https://img.shields.io/badge/Apple-MLX-black)](https://github.com/Scottcjn/mlx-power8) [![POWER8](https://img.shields.io/badge/IBM-POWER8-red)](https://github.com/Scottcjn/mlx-power8)
[![BCOS Certified](https://img.shields.io/badge/BCOS-Certified-brightgreen?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0xMiAxTDMgNXY2YzAgNS41NSAzLjg0IDEwLjc0IDkgMTIgNS4xNi0xLjI2IDktNi40NSA5LTEyVjVsLTktNHptLTIgMTZsLTQtNCA1LjQxLTUuNDEgMS40MSAxLjQxTDEwIDE0bDYtNiAxLjQxIDEuNDFMMTAgMTd6Ii8+PC9zdmc+)](BCOS.md)

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
