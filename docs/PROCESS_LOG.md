# Process log

Dated record of what was done, what changed from `PLAN.md` and why. Plan changes are recorded here rather than silently edited into the plan.

## 2026-10-05

- **PLAN.md committed** before any implementation (commit `bd42450`).
- **Section 1: project setup.** `pyproject.toml` (Python ≥3.12), `fleet_signal` package skeleton, GitHub Actions CI running `ruff check`, `ruff format --check` and `pytest` on Python 3.12. The package is `fleet_signal`, not `signal`, because `signal` would shadow the standard-library module.
