# Contributing to mlx-power8

Thanks for contributing to the MLX Linux POWER8 and x86 SIMD port. Changes here
touch low-level vectorization paths, so correctness, architecture detection, and
small reproducible tests are especially important.

## Useful Contributions

- Improve VSX, AVX, or architecture-detection documentation.
- Add compile checks for POWER8, POWER9, or x86_64 environments.
- Fix portability issues in SIMD headers.
- Document integration steps for newer MLX source layouts.
- Add small examples that exercise vector operations consistently.

## Development Workflow

1. Fork the repository and create a focused branch.
2. Keep VSX, AVX, and documentation changes separate when practical.
3. Include compiler, architecture, flags, and host OS in the PR description.
4. Avoid broad formatting changes that make SIMD diffs hard to review.

## Validation

- Documentation-only changes: run `git diff --check`.
- Header changes: compile a minimal translation unit that includes the touched
  header on the target architecture.
- Integration changes: copy the files into the intended MLX source tree and
  report the build command used.

Example compile check:

```bash
cat > check.cpp << 'EOF'
#include "simd/vsx_simd.h"
int main() { return 0; }
EOF
c++ -std=c++17 -c check.cpp
```

## Pull Request Checklist

- The affected architecture is named.
- Compiler and build flags are included.
- SIMD behavior changes include a correctness check.
- Compatibility with existing MLX file layout is described.
- Generated build outputs are not committed.

## Reporting Issues

Include CPU architecture, compiler version, operating system, MLX version or
commit if applicable, the command that failed, and the full compiler output.
