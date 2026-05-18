# Contributing to mlx-power8

Thanks for helping with mlx-power8. This repository carries POWER8 SIMD support
for MLX, so contributions should stay tightly scoped and easy to transplant
into an MLX source tree.

## Setup

1. Fork the repository and create a branch:

   ```sh
   git checkout -b fix/short-description
   ```

2. Review `README.md` and `BUILD_POWER8.md` before changing the SIMD files.

3. Copy the `simd/` contents into the matching MLX source tree location when
   testing integration:

   ```sh
   cp -R simd/* /path/to/mlx/mlx/backend/cpu/simd/
   ```

4. Build MLX on a POWER8 or compatible ppc64le host when hardware is available.
   If hardware is not available, state that clearly in the PR and run the checks
   that can be done locally.

## Pull Request Guidelines

- Keep one architecture or build-system change per PR.
- Describe the POWER8, compiler, and operating system used for validation.
- Include the exact MLX tree or commit used for integration testing.
- Update `BUILD_POWER8.md` when build flags, dependencies, or copy paths change.
- Do not commit generated build outputs.

## Code Style

- Preserve the existing SIMD file organization.
- Keep POWER8-specific code isolated and clearly named.
- Prefer compiler feature checks or documented build flags over hidden
  assumptions.
- Use comments only for non-obvious vector behavior or architecture constraints.
- Avoid changes that alter non-POWER behavior unless the PR explicitly covers
  that compatibility.

## Validation Checklist

- [ ] `simd/` files were copied into an MLX tree or the limitation is explained.
- [ ] POWER8/ppc64le build was run when hardware was available.
- [ ] Build flags and compiler versions are listed in the PR.
- [ ] Documentation was updated for setup or behavior changes.
- [ ] Generated files and local build artifacts are excluded.
