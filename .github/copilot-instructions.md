# AI Agent Context for JFM_CS

## Project Overview

This repository contains a scientific Python pipeline for spectral analysis and wake dynamics around bluff-body geometries.
Core scientific modules are in `core/`; the DSGBR package is in `dsgbr/`.

## Build and Environment

- Use `uv` for all package and environment operations.
- For quick Python checks: `uv run pytest`.
- Package build: `uv build`.
- Distribution validation: `uv run twine check dist/*`.

## Quality Standards

- Lint, format, and static checks are defined via `.pre-commit-config.yaml` and `.github/workflows/ci.yml`.
- Prefer small, deterministic changes to analysis logic.
- Maintain reproducibility in helper scripts and test data assumptions.

## Testing Conventions

- Add regression tests for algorithm behavior changes.
- Keep tests deterministic and fast where possible.
- No test weakening to satisfy existing behavior.

## Documentation

- Keep scientific assumptions and data assumptions explicit in docs and comments.
- Prefer short, auditable scripts with clear CLI examples in `README.md`.
