# Building MLX on POWER8 Linux

## Prerequisites
```bash
# Ubuntu 20.04 on POWER8 (ppc64le)
sudo apt install build-essential cmake python3-dev python3-pip
```

## Build Steps

```bash
# Clone MLX with POWER8 support
git clone https://github.com/Scottcjn/mlx-power8.git
cd mlx-power8

# Create build directory
mkdir build && cd build

# Configure - CPU backend only (no Metal on Linux)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLX_BUILD_CPU=ON \
    -DMLX_BUILD_METAL=OFF \
    -DMLX_BUILD_CUDA=OFF \
    -DCMAKE_CXX_FLAGS="-mcpu=power8 -mvsx -O3"

# Build
make -j$(nproc)

# Install Python bindings
pip install .
```

## POWER8 VSX Optimizations

The `vsx_simd.h` provides POWER8-optimized SIMD:
- 4-wide float vectors (128-bit VSX)
- 2-wide double vectors
- Hardware FMA (fused multiply-add)
- Hardware rsqrt estimate

## Running exo with MLX on POWER8

```bash
# Start exo node on POWER8
exo run --inference-engine mlx

# Connect to x86 cluster
exo connect <x86_node_ip>:5678
```
