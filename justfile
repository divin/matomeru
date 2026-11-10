set dotenv-load := false

# Default: format, lint, and sort imports.
default: fmt lint isort

# Format the code.
fmt:
    uv run ruff format src tests

# Lint and automatically fix safe issues.
lint:
    uv run ruff check --fix src tests

# Sort Python imports with Ruff.
isort:
    uv run ruff check --select I --fix src tests

# Run non-mutating checks.
check:
    uv run ty check .
    uv run ruff format --check src tests
    uv run ruff check --extend-ignore I src tests
    uv run ruff check --select I src tests

# Run the test suite.
test:
    uv run pytest tests -q

# Install/update the locked Python environment.
sync:
    uv sync --dev
