# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Centralized configuration system using Pydantic Settings (`urcm/config.py`)
- `.env.example` documenting all environment variables
- CHANGELOG.md

### Changed
- Migrated all env var access to centralized config module
- `api.py`, `safety.py`, `observability.py`, `metrics_exporter.py` now use `get_settings()`

### Fixed
- TruthMaintenanceSystem extracted to its own module (`urcm/core/tms.py`)
- Indentation error in `reasoning.py` (lines 64-76)

## [0.1.0] - 2025-01-18

### Added
- Core resonance engine with hierarchical encoding
- FastAPI-based API server with async support
- Docker and Kubernetes deployment configurations
- CI/CD pipeline with linting, testing, and integration tests
- Locust load testing infrastructure
- Integration test suite with test data fixtures
- Observability module with structured event logging and log rotation
- Safety Governor with energy ceiling, kernel lock, and reversibility checks
- Truth Maintenance System for logical consistency
- A* planning with SMT/Z3 constraint solving
- Executive Controller with goal-driven reasoning
- Convergence Engine with mu-stability convergence detection
- Memory subsystem with bounded geometric memory deposition
- Performance benchmarking and compression monitoring
- Prometheus metrics export
- Health and validation endpoints

### Security
- Replaced pickle deserialization with safe serialization
- Environment-based credential management
- Bearer token authentication on API endpoints
- Input validation with Pydantic models
- CORS configuration

### Fixed
- Bare except clauses replaced with specific exception types
- Graceful shutdown handlers
- Audit failure alerting
