# Contributing to mlx-power8

Thank you for your interest in contributing to mlx-power8! This project brings Apple's MLX machine learning framework to IBM POWER8 architecture, enabling ML workloads on PowerPC systems.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project adheres to a code of conduct that expects all participants to:
- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

## Getting Started

### Prerequisites

To contribute to mlx-power8, you'll need:

- **Hardware**: IBM POWER8 or POWER9 system (or QEMU ppc64le emulation)
- **Operating System**: Linux (Ubuntu 18.04+, RHEL 7+, or Debian 9+)
- **Compiler**: GCC 8+ or Clang 9+ with POWER8 VSX support
- **Python**: 3.8+ (for Python bindings)
- **CMake**: 3.14+
- **Git**: 2.20+

### Repository Structure

```
mlx-power8/
├── src/              # Core C++ implementation
├── python/           # Python bindings
├── tests/            # Test suite
├── docs/             # Documentation
├── examples/         # Example code
├── benchmarks/       # Performance benchmarks
└── scripts/          # Build and utility scripts
```

## Development Environment

### Setting Up QEMU (for non-POWER8 developers)

If you don't have physical POWER8 hardware, use QEMU for development:

```bash
# Install QEMU
sudo apt-get install qemu-system-ppc qemu-user-static

# Download ppc64le Ubuntu cloud image
wget https://cloud-images.ubuntu.com/minimal/releases/jammy/release/ubuntu-22.04-minimal-cloudimg-ppc64el.img

# Create VM with sufficient resources
qemu-system-ppc64 -m 4096 -smp 4 -cpu POWER8 \
  -drive file=ubuntu-22.04-minimal-cloudimg-ppc64el.img,format=raw \
  -netdev user,id=net0 -device virtio-net-pci,netdev=net0
```

### Native POWER8 Development

On a native POWER8 system:

```bash
# Clone the repository
git clone https://github.com/Scottcjn/mlx-power8.git
cd mlx-power8

# Create build directory
mkdir build && cd build

# Configure with POWER8 optimizations
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_C_FLAGS="-mcpu=power8 -mtune=power8" \
         -DCMAKE_CXX_FLAGS="-mcpu=power8 -mtune=power8"

# Build
make -j$(nproc)

# Run tests
ctest --output-on-failure
```

## How to Contribute

### Reporting Bugs

Before reporting a bug:
1. Check existing issues to avoid duplicates
2. Test on the latest main branch
3. Gather system information

When reporting, include:
- POWER8/POWER9 system details (`lscpu` output)
- Operating system and version
- Compiler version (`gcc --version`)
- Python version (if applicable)
- Minimal code to reproduce the issue
- Expected vs actual behavior
- Error messages and stack traces

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:
- Explain the use case
- Describe the proposed solution
- Consider backward compatibility
- Discuss potential performance implications

### Contributing Code

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'feat: add amazing feature'`)
4. **Push** to your fork (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

## Coding Standards

### C++ Guidelines

Follow these POWER8-specific conventions:

```cpp
// Use VSX intrinsics for vector operations
#include <altivec.h>

// Prefer explicit vector types
vector float vec_add(vector float a, vector float b) {
    return vec_add(a, b);  // VSX intrinsic
}

// Align data for vector loads
alignas(16) float data[4];

// Use restrict for optimization hints
void compute(float* __restrict__ dst, 
             const float* __restrict__ src, size_t n);
```

### Performance Considerations

- **Vectorization**: Use VSX (Vector Scalar Extension) for SIMD operations
- **Cache awareness**: POWER8 has 64-byte cache lines - align accordingly
- **Memory ordering**: Use appropriate memory barriers for atomic operations
- **NUMA awareness**: Consider memory placement on multi-socket systems

### Python Bindings

```python
# Follow PEP 8 style
# Use type hints where appropriate
def matmul(a: mx.array, b: mx.array) -> mx.array:
    """Matrix multiplication optimized for POWER8."""
    return mx.matmul(a, b)
```

## Testing

### Running Tests

```bash
# Build with testing enabled
cmake .. -DMLX_BUILD_TESTS=ON
make -j$(nproc)

# Run all tests
ctest --output-on-failure

# Run specific test
ctest -R test_power8_ops -V
```

### Writing Tests

Add tests for new functionality:

```cpp
// tests/test_power8_ops.cpp
TEST(Power8Ops, VectorAdd) {
    alignas(16) float a[4] = {1, 2, 3, 4};
    alignas(16) float b[4] = {5, 6, 7, 8};
    alignas(16) float c[4];
    
    vector_add_power8(a, b, c, 4);
    
    EXPECT_FLOAT_EQ(c[0], 6.0f);
    EXPECT_FLOAT_EQ(c[1], 8.0f);
    EXPECT_FLOAT_EQ(c[2], 10.0f);
    EXPECT_FLOAT_EQ(c[3], 12.0f);
}
```

### Benchmarking

Performance changes must include benchmarks:

```bash
# Run benchmarks
./benchmarks/bench_power8_ops --benchmark_format=csv > results.csv

# Compare against baseline
./scripts/compare_benchmarks.py baseline.csv results.csv
```

## Documentation

### Code Documentation

Use Doxygen-style comments:

```cpp
/**
 * @brief Perform optimized matrix multiplication on POWER8
 * 
 * Uses VSX instructions to accelerate 4x4 matrix blocks.
 * Input matrices must be aligned to 16-byte boundaries.
 * 
 * @param A Left matrix (M x K)
 * @param B Right matrix (K x N)
 * @param C Output matrix (M x N)
 * @param M Rows in A and C
 * @param K Columns in A / rows in B
 * @param N Columns in B and C
 * @return 0 on success, non-zero on error
 */
int matmul_power8(const float* A, const float* B, float* C,
                  size_t M, size_t K, size_t N);
```

### README Updates

Update README.md if your change affects:
- Build instructions
- Dependencies
- API changes
- Performance characteristics

## Commit Guidelines

Use conventional commits:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `style:` - Formatting, semicolons, etc.
- `refactor:` - Code restructuring
- `perf:` - Performance improvement
- `test:` - Adding or correcting tests
- `chore:` - Build process, dependencies

Examples:
```
feat: add VSX-optimized convolution
fix: correct alignment check in matmul
docs: update POWER8 build instructions
perf: improve cache utilization in gemm
```

## Pull Request Process

1. **Update documentation** for any API changes
2. **Add tests** for new functionality
3. **Run the test suite** and ensure all tests pass
4. **Update CHANGELOG.md** with your changes
5. **Ensure your PR**:
   - Has a clear description
   - References related issues
   - Includes benchmark results for performance changes
   - Has been tested on actual POWER8 hardware (if possible)

### PR Review Criteria

Maintainers will review for:
- Code correctness
- POWER8 optimization effectiveness
- Test coverage
- Documentation completeness
- Backward compatibility

## POWER8-Specific Resources

- [POWER8 Processor Manual](https://ibm.biz/power8-manual)
- [VSX Programming Guide](https://ibm.biz/vsx-guide)
- [POWER8 Optimization Guide](https://ibm.biz/power8-opt)
- [ppc64le ABI](https://ibm.biz/ppc64le-abi)

## Community

- **Issues**: GitHub Issues for bug reports and features
- **Discussions**: GitHub Discussions for questions
- **Chat**: Matrix channel #