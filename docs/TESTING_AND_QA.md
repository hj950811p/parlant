# Testing and Quality Assurance Guide

## Overview

This document provides a comprehensive guide to the testing and quality assurance infrastructure for the Parlant framework. It covers test organization, fixtures, running tests, and troubleshooting.

## Quick Links

- **[Testing Guide](./advanced/testing-guide.md)** - How to write and run tests
- **[Test Fixtures Guide](./advanced/test-fixtures.md)** - Fixture patterns and best practices
- **[Cerebras Adapter Documentation](./adapters/nlp/cerebras.md)** - NLP service adapter reference

## What's Tested

### 1. NLP Service Adapters

**Location**: `tests/adapters/nlp/`

Comprehensive testing of NLP service implementations including:
- **Cerebras Service** (`test_cerebras_service.py`)
  - Model initialization and configuration
  - Token estimation
  - Schema-based generation
  - Error handling (rate limits, timeouts, connection errors)
  - JSON validation and normalization

- **Common Utilities** (`test_common.py`)
  - JSON normalization
  - LLM metrics recording

- **Integration Tests** (`test_nlp_integration.py`)
  - End-to-end generation workflows
  - Multiple sequential operations
  - Custom hints and parameters
  - Error recovery

### 2. Core Framework

**Location**: `tests/core/`

Stable core components and unstable experimental features with unit and integration tests covering:
- Entity operations
- Guidelines and journeys
- Context variables
- Tool calling
- Message generation

### 3. API Endpoints

**Location**: `tests/api/`

REST API endpoint testing with:
- Request validation
- Response serialization
- Error handling

### 4. SDK Integration

**Location**: `tests/sdk/`

High-level SDK integration tests covering:
- Agent creation and management
- Guideline application
- Variable management
- Tool integration

### 5. End-to-End Scenarios

**Location**: `tests/e2e/`

Complete user workflows including:
- Server CLI
- Client CLI integration
- Multi-turn conversations

## Test Structure

### Unit Tests

Test individual components in isolation:

```python
# tests/adapters/nlp/test_cerebras_service.py
class TestCerebrasSchematicGenerator:
    def test_initialization(self, mock_logger, mock_meter):
        """Test generator initialization."""
        generator = CerebrasSchematicGenerator(...)
        assert generator.model_name == "llama3.1-8b"
```

### Integration Tests

Test components working together:

```python
# tests/adapters/nlp/test_nlp_integration.py
class TestCerebrasIntegration:
    @pytest.mark.asyncio
    async def test_end_to_end_generation(self, mock_logger, mock_meter):
        """Test full generation pipeline."""
        service = CerebrasService(mock_logger, mock_meter)
        generator = await service.get_schematic_generator(Schema)
        result = await generator.do_generate(prompt)
        assert result is not None
```

## Test Fixtures

### Common Fixtures

Available in all tests via `tests/conftest.py`:

- `mock_logger` - Mocked Logger instance
- `mock_meter` - Mocked Meter instance
- `user_profile_schema` - Sample Pydantic schema
- `response_schema` - Sample response schema

### Cerebras-Specific Fixtures

Available in `tests/conftest_cerebras.py`:

- `mock_api_response_success` - Successful API response
- `mock_api_response_empty_content` - Empty response
- `mock_api_response_invalid_json` - Invalid JSON response
- `rate_limit_error` - RateLimitError
- `connection_error` - APIConnectionError
- `timeout_error` - APITimeoutError

### Using Fixtures

```python
@pytest.mark.asyncio
async def test_generation(mock_logger, mock_meter, mock_api_response_success):
    """Test with multiple fixtures."""
    generator = Llama3_3_70B(mock_logger, mock_meter)
    generator._client = AsyncMock()
    generator._client.chat.completions.create = AsyncMock(
        return_value=mock_api_response_success
    )
    result = await generator._do_generate("prompt")
    assert result.content is not None
```

## Running Tests

### All Tests

```bash
# Run all tests
poetry run pytest tests/

# Run with verbose output
poetry run pytest tests/ -v

# Run with coverage
poetry run pytest tests/ --cov=src/parlant
```

### Specific Test Categories

```bash
# NLP adapter tests
poetry run pytest tests/adapters/nlp/

# Cerebras tests
poetry run pytest tests/adapters/nlp/test_cerebras_service.py

# Single test
poetry run pytest tests/adapters/nlp/test_cerebras_service.py::TestCerebrasSchematicGenerator::test_initialization
```

### Integration Tests

```bash
# All integration tests
poetry run pytest tests/adapters/nlp/test_nlp_integration.py

# Specific integration test
poetry run pytest tests/adapters/nlp/test_nlp_integration.py::TestCerebrasIntegration::test_end_to_end_generation
```

### E2E Tests

```bash
# All end-to-end tests
poetry run pytest tests/e2e/

# With server running
poetry run pytest tests/e2e/ -v
```

## Test Coverage

### Coverage Goals

- Adapters: Minimum 80%
- Core engines: Minimum 75%
- API endpoints: Minimum 85%
- SDK: Minimum 80%

### Check Coverage

```bash
# Generate coverage report
poetry run pytest tests/ --cov=src/parlant --cov-report=html

# View report
open htmlcov/index.html
```

## CI/CD Integration

Tests run automatically in GitHub Actions:

```yaml
# .github/workflows/ci-test.yml
- name: Test Parlant (deterministic)
  run: just test-deterministic

- name: Test Parlant (core-stable)
  run: just test-core-stable

- name: Test Parlant (core-unstable)
  run: just test-core-unstable
```

## Type Checking

### MyPy Strict Mode

All code must pass MyPy strict type checking:

```bash
# Check types
poetry run mypy

# Check specific file
poetry run mypy src/parlant/adapters/nlp/cerebras_service.py
```

### Type Hints

All public functions and classes must have:
- Parameter type annotations
- Return type annotations
- Docstring type hints

Example:
```python
async def generate(
    self,
    prompt: str,
    hints: Mapping[str, Any] = {},
) -> SchematicGenerationResult[T]:
    """Generate structured output.
    
    Args:
        prompt: Input prompt text.
        hints: Generation hints (temperature, etc).
        
    Returns:
        SchematicGenerationResult with generated content and metadata.
    """
```

## Code Style

### Formatting

All code must be formatted with Ruff:

```bash
# Format code
poetry run ruff format

# Check formatting
poetry run ruff format --check
```

### Linting

Code must pass Ruff linting:

```bash
# Check linting
poetry run ruff check

# Fix linting issues
poetry run ruff check --fix
```

## Documentation

### Docstrings

All public classes and functions should have docstrings:

```python
class CerebrasSchematicGenerator(BaseSchematicGenerator[T]):
    """Generic schematic generator for Cerebras Llama models.
    
    This generator uses Cerebras' high-performance inference API...
    """
    
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        """Generate structured output from a prompt.
        
        Args:
            prompt: The input prompt (string or PromptBuilder).
            hints: Optional generation hints like temperature.
            
        Returns:
            A SchematicGenerationResult containing the structured output.
            
        Raises:
            RateLimitError: If API rate limits are exceeded.
            ValidationError: If generated content doesn't match schema.
        """
```

### Module Documentation

Each module should have a module-level docstring:

```python
"""NLP service adapter implementations.

This module provides adapters for various NLP services including Cerebras,
OpenAI, Anthropic, and others. Each adapter implements the NLPService interface
and provides schema-based generation, embeddings, and moderation services.
"""
```

## Troubleshooting

### Test Failures

**Mocking issues**: Verify mock setup matches actual API changes
```python
# Check if mock attributes match the real implementation
generator._client = AsyncMock()
generator._client.chat.completions.create = AsyncMock(...)
```

**Async/await errors**: Ensure `@pytest.mark.asyncio` is used
```python
@pytest.mark.asyncio  # Required for async tests
async def test_async_function(mock_logger):
    result = await some_async_function()
```

**Import errors**: Check Python path and package installation
```bash
# Verify package is installed
poetry install

# Check Python path
python -c "import parlant; print(parlant.__file__)"
```

### Slow Tests

Profile with pytest-duration:
```bash
poetry run pytest tests/ --durations=10
```

Parallelize with pytest-xdist:
```bash
poetry run pytest tests/ -n auto
```

## Best Practices

1. **Isolation**: Each test should be independent
2. **Clarity**: Use descriptive test names
3. **Mocking**: Mock external dependencies
4. **Assertions**: Use specific assertions
5. **Performance**: Keep tests under 100ms each
6. **Documentation**: Document complex test setups

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [MyPy](https://www.mypy-lang.org/)
- [Ruff](https://docs.astral.sh/ruff/)

## Contributing

When adding new functionality:

1. Write tests first (TDD)
2. Ensure tests pass
3. Add type hints to all functions
4. Add comprehensive docstrings
5. Update documentation
6. Run linting/formatting/type checking
7. Submit PR with tests

For more details, see [CONTRIBUTING.md](../CONTRIBUTING.md).
