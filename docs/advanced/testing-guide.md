# Testing Guide

## Overview

Parlant uses a comprehensive testing strategy including unit tests, integration tests, and end-to-end tests. This guide covers testing best practices for the framework.

## Test Structure

Tests are organized to mirror the source code structure:

```
tests/
├── adapters/          # Adapter tests (NLP, vector DB, document stores)
├── api/               # REST API endpoint tests  
├── core/              # Core framework tests
├── e2e/               # End-to-end scenarios
├── modules/           # Test modules and fixtures
└── sdk/               # SDK integration tests
```

The equivalent source code structure is:

```
src/parlant/
├── adapters/          # NLP, persistence, and tool adapters
├── api/               # FastAPI endpoints
├── core/              # Domain logic and engines
├── bin/               # CLI commands
└── sdk.py             # Main SDK interface
```

## Test Types

### Unit Tests

Unit tests verify individual components in isolation using mocks for dependencies.

**Location**: `tests/adapters/`, `tests/core/`

**Example**:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_generation_success(mock_logger, mock_meter):
    """Test successful generation."""
    generator = Llama3_3_8B(mock_logger, mock_meter)
    
    # Mock API response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"name": "test"}'
    
    generator._client = AsyncMock()
    generator._client.chat.completions.create = AsyncMock(
        return_value=mock_response
    )
    
    result = await generator._do_generate("prompt")
    assert result.content.name == "test"
```

### Integration Tests

Integration tests verify components working together with external services mocked.

**Location**: `tests/adapters/nlp/test_nlp_integration.py`

**Example**:
```python
@pytest.mark.asyncio
async def test_end_to_end_generation(mock_logger, mock_meter):
    """Test full generation pipeline."""
    service = CerebrasService(mock_logger, mock_meter)
    generator = await service.get_schematic_generator(UserSchema)
    
    # Mock and verify the full pipeline
    result = await generator.do_generate("prompt", {"temperature": 0.5})
    assert isinstance(result.info, GenerationInfo)
```

### End-to-End Tests

End-to-end tests verify complete user scenarios with realistic setups.

**Location**: `tests/e2e/`

**Example**:
```python
@pytest.mark.asyncio
async def test_agent_conversation_flow(server):
    """Test complete agent conversation."""
    agent = await server.create_agent(name="TestAgent")
    
    # Verify full conversation flow
    response = await agent.chat("Hello")
    assert response is not None
```

## Running Tests

### Run All Tests

```bash
# Run all tests
poetry run pytest tests/

# Run with verbose output
poetry run pytest tests/ -v

# Run with coverage
poetry run pytest tests/ --cov=src/parlant
```

### Run Specific Test Categories

```bash
# Run adapter tests
poetry run pytest tests/adapters/

# Run NLP adapter tests
poetry run pytest tests/adapters/nlp/

# Run Cerebras tests
poetry run pytest tests/adapters/nlp/test_cerebras_service.py

# Run single test
poetry run pytest tests/adapters/nlp/test_cerebras_service.py::TestCerebrasSchematicGenerator::test_initialization
```

### Run Deterministic Tests

```bash
# Tests that don't require external services
poetry run pytest tests/ -m deterministic
```

## Writing Tests

### Setup Fixtures

Fixtures are reusable test components:

```python
import pytest
from unittest.mock import MagicMock
from parlant.core.loggers import Logger
from parlant.core.meter import Meter

@pytest.fixture
def mock_logger() -> Logger:
    """Reusable mock logger fixture."""
    logger = MagicMock(spec=Logger)
    logger.scope = MagicMock()
    logger.scope.return_value.__enter__ = MagicMock(return_value=None)
    logger.scope.return_value.__exit__ = MagicMock(return_value=None)
    return logger

@pytest.fixture
def mock_meter() -> Meter:
    """Reusable mock meter fixture."""
    return MagicMock(spec=Meter)
```

### Testing Async Code

All async functions require the `@pytest.mark.asyncio` decorator:

```python
@pytest.mark.asyncio
async def test_async_function(mock_logger):
    """Test an async function."""
    result = await some_async_function()
    assert result is not None
```

### Mocking External Services

Mock external API calls to avoid dependencies:

```python
from unittest.mock import AsyncMock, Mock

@pytest.mark.asyncio
async def test_with_mocked_api(generator):
    """Test with mocked API response."""
    # Setup mock response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"field": "value"}'
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    
    # Replace client method
    generator._client = AsyncMock()
    generator._client.chat.completions.create = AsyncMock(
        return_value=mock_response
    )
    
    # Verify behavior with mocked service
    result = await generator._do_generate("prompt")
    assert result.info.usage.input_tokens == 10
```

### Testing Error Handling

Test that errors are handled correctly:

```python
import pytest

@pytest.mark.asyncio
async def test_rate_limit_error(generator):
    """Test handling of rate limit errors."""
    from cerebras.cloud.sdk import RateLimitError
    
    generator._client = AsyncMock()
    generator._client.chat.completions.create = AsyncMock(
        side_effect=RateLimitError("Rate limited")
    )
    
    with pytest.raises(Exception):
        await generator._do_generate("prompt")
```

## Test Coverage

### Coverage Goals

- **Adapters**: Minimum 80% coverage
- **Core engines**: Minimum 75% coverage  
- **API endpoints**: Minimum 85% coverage
- **SDK**: Minimum 80% coverage

### Check Coverage

```bash
# Generate coverage report
poetry run pytest tests/ --cov=src/parlant --cov-report=html

# View in browser
open htmlcov/index.html
```

## Best Practices

1. **Isolation**: Each test should be independent and repeatable
2. **Clarity**: Use descriptive test names explaining what is tested and expected
3. **Mocking**: Mock external dependencies to avoid flaky tests
4. **Assertions**: Use specific assertions with clear error messages
5. **Performance**: Keep tests fast (< 100ms per test if possible)
6. **Documentation**: Document complex test setups with comments

## Example Test Files

### Comprehensive Unit Test

See: `tests/adapters/nlp/test_cerebras_service.py`

This file demonstrates:
- Fixture usage
- Mocking API responses
- Error handling tests
- Property testing
- Initialization tests

### Comprehensive Integration Test

See: `tests/adapters/nlp/test_nlp_integration.py`

This file demonstrates:
- End-to-end generation workflows
- Error handling across components
- Custom hints and parameters
- Service initialization
- Sequential operations

## Continuous Integration

Tests are automatically run in CI/CD:

```yaml
# .github/workflows/ci-test.yml
- name: Test Parlant (deterministic)
  run: just test-deterministic

- name: Test Parlant (core-stable)
  run: just test-core-stable
```

All tests must pass before merging.

## Troubleshooting

### Test Failures

1. **Check mocks**: Verify mock setup matches actual API changes
2. **Async issues**: Ensure `@pytest.mark.asyncio` is used
3. **Timing**: Some tests may be flaky; add retries if needed
4. **Environment**: Ensure test environment variables are set

### Slow Tests

1. Profile with `pytest-duration` to identify slow tests
2. Move integration tests to separate suite
3. Parallelize with `pytest-xdist`: `pytest tests/ -n auto`

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [unittest.mock Guide](https://docs.python.org/3/library/unittest.mock.html)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
