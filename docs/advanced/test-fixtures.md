# Test Fixtures and Golden Files

## Overview

Test fixtures provide reusable test data and setup for consistent testing across the Parlant codebase. This guide covers fixture organization, usage, and best practices.

## Fixture Organization

### Fixture Locations

Fixtures are organized by scope and availability:

1. **Global Fixtures** - Available to all tests
   - Location: `tests/conftest.py` (main)
   - Location: `tests/conftest_cerebras.py` (Cerebras-specific)

2. **Directory Fixtures** - Available within a directory
   - Location: `tests/adapters/conftest.py`
   - Location: `tests/core/conftest.py`

3. **Module Fixtures** - Specific to a test file
   - Defined directly in test files

### Fixture Scope

```python
@pytest.fixture(scope="function")  # Default - new instance per test
def mock_logger():
    return MagicMock()

@pytest.fixture(scope="session")  # Reused across all tests
def shared_client():
    return SomeExpensiveResource()

@pytest.fixture(scope="module")   # Reused within a module
def shared_config():
    return load_config()
```

## Core Fixtures

### Logger and Meter

The most commonly used fixtures for Parlant testing:

```python
# From tests/conftest_cerebras.py

@pytest.fixture
def mock_logger() -> Logger:
    """Mock logger for testing."""
    logger = MagicMock(spec=Logger)
    logger.scope = MagicMock()
    logger.scope.return_value.__enter__ = MagicMock(return_value=None)
    logger.scope.return_value.__exit__ = MagicMock(return_value=None)
    logger.info = MagicMock()
    logger.error = MagicMock()
    return logger

@pytest.fixture
def mock_meter() -> Meter:
    """Mock meter for recording metrics."""
    meter = MagicMock(spec=Meter)
    meter.record_value = AsyncMock()
    return meter
```

### Schema Fixtures

Mock Pydantic models for testing generation:

```python
@pytest.fixture
def user_profile_schema() -> type[MockUserProfile]:
    """Mock user profile schema."""
    return MockUserProfile

@pytest.fixture
def response_schema() -> type[MockResponseSchema]:
    """Mock response schema."""
    return MockResponseSchema
```

## API Response Fixtures

### Successful Response

```python
@pytest.fixture
def mock_api_response_success() -> Any:
    """Mock successful Cerebras API response."""
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = '''{
        "username": "test_user",
        "email": "test@example.com",
        "age": 30
    }'''
    response.usage = Mock()
    response.usage.prompt_tokens = 50
    response.usage.completion_tokens = 30
    return response
```

### Error Responses

```python
@pytest.fixture
def mock_api_response_empty_content() -> Any:
    """Mock response with no content."""
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = None
    return response

@pytest.fixture
def mock_api_response_invalid_json() -> Any:
    """Mock response with invalid JSON."""
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = "Not valid JSON"
    return response
```

## Error Fixtures

Fixtures for testing error scenarios:

```python
@pytest.fixture
def rate_limit_error():
    """RateLimitError for testing."""
    from cerebras.cloud.sdk import RateLimitError
    return RateLimitError("Rate limit exceeded")

@pytest.fixture
def connection_error():
    """APIConnectionError for testing."""
    from cerebras.cloud.sdk import APIConnectionError
    return APIConnectionError("Connection failed")

@pytest.fixture
def timeout_error():
    """APITimeoutError for testing."""
    from cerebras.cloud.sdk import APITimeoutError
    return APITimeoutError("Request timed out")
```

## Using Fixtures in Tests

### Basic Usage

```python
def test_logger_initialization(mock_logger):
    """Test that logger is properly initialized."""
    assert mock_logger is not None
    mock_logger.info("test message")
    mock_logger.info.assert_called_once()
```

### Multiple Fixtures

```python
@pytest.mark.asyncio
async def test_generation(mock_logger, mock_meter, mock_api_response_success):
    """Test generation with multiple fixtures."""
    generator = Llama3_3_70B(mock_logger, mock_meter)
    generator._client = AsyncMock()
    generator._client.chat.completions.create = AsyncMock(
        return_value=mock_api_response_success
    )
    
    result = await generator._do_generate("prompt")
    assert result is not None
```

### Fixture Dependencies

```python
@pytest.fixture
def configured_generator(mock_logger, mock_meter):
    """Generator configured with mocks."""
    return Llama3_3_70B(mock_logger, mock_meter)

@pytest.mark.asyncio
async def test_with_configured_generator(configured_generator):
    """Test using pre-configured generator."""
    # Generator is already initialized
    assert configured_generator is not None
```

## Golden Files

Golden files are reference outputs used to verify correct behavior and detect regressions.

### Golden File Organization

```
tests/
├── data/
│   ├── cerebras/
│   │   ├── generation_output_user_profile.json
│   │   ├── generation_output_response.json
│   │   └── tokenizer_estimates.json
│   └── nlp/
│       └── normalized_json_samples.json
└── fixtures/
    └── cerebras/
        ├── api_responses/
        │   ├── success_response.json
        │   └── error_response.json
        └── schemas/
            ├── user_profile.json
            └── response_schema.json
```

### Creating Golden Files

```python
# Save golden file during test development
import json

def test_generation_produces_valid_output(generator):
    """Test and save golden file."""
    result = await generator._do_generate("prompt")
    
    # On first run, save as golden file
    golden_path = "tests/data/cerebras/generation_output.json"
    with open(golden_path, "w") as f:
        json.dump(result.model_dump(), f, indent=2)
```

### Using Golden Files for Regression Testing

```python
import json

def test_generation_matches_golden_file(generator, mock_api_response_success):
    """Verify generation matches known-good output."""
    generator._client = AsyncMock()
    generator._client.chat.completions.create = AsyncMock(
        return_value=mock_api_response_success
    )
    
    result = await generator._do_generate("prompt")
    
    # Load golden file
    golden_path = "tests/data/cerebras/generation_output.json"
    with open(golden_path) as f:
        golden_data = json.load(f)
    
    # Verify matches
    assert result.model_dump() == golden_data
```

### Golden File Best Practices

1. **Version Control**: Commit golden files to git
2. **Review Changes**: Review diffs when golden files change
3. **Clear Purpose**: Name files descriptively
4. **Documentation**: Comment what each golden file tests
5. **Update Carefully**: Only update after verifying changes are correct

## Example Fixtures in Tests

### Cerebras Service Tests

Location: `tests/adapters/nlp/test_cerebras_service.py`

This file uses fixtures like:
- `mock_logger` - For logging
- `mock_meter` - For metrics
- Generated fixtures for schemas

### NLP Integration Tests

Location: `tests/adapters/nlp/test_nlp_integration.py`

This file uses:
- `mock_logger` - Logging
- `mock_meter` - Metrics
- Pydantic model fixtures
- Mock API response fixtures

## Fixture Parameters

Fixtures can be parametrized to test multiple scenarios:

```python
@pytest.fixture(params=["8B", "70B"])
def model_size(request):
    """Parametrized fixture for model sizes."""
    return request.param

def test_all_model_sizes(model_size):
    """Test runs with both 8B and 70B."""
    if model_size == "8B":
        generator = Llama3_3_8B(mock_logger, mock_meter)
    else:
        generator = Llama3_3_70B(mock_logger, mock_meter)
    assert generator is not None
```

## Fixture Utilities

### Creating Reusable Fixture Builders

```python
def create_mock_logger(name="test"):
    """Builder function for creating custom loggers."""
    logger = MagicMock(spec=Logger)
    logger.name = name
    return logger

def create_mock_response(success=True, message="OK"):
    """Builder for creating mock responses."""
    response = Mock()
    response.success = success
    response.message = message
    return response
```

## Testing with Fixtures

### Common Patterns

1. **Setup + Assertion**
   ```python
   def test_something(mock_logger):
       """Simple test pattern."""
       obj = SomeClass(mock_logger)
       assert obj is not None
   ```

2. **Mock Behavior**
   ```python
   def test_mocked_call(mock_logger):
       """Test with mocked behavior."""
       obj = SomeClass(mock_logger)
       mock_logger.info.return_value = "value"
       result = obj.call_method()
       mock_logger.info.assert_called()
   ```

3. **Async Operations**
   ```python
   @pytest.mark.asyncio
   async def test_async_operation(mock_meter):
       """Test async operation."""
       await mock_meter.record_value("key", 100)
       mock_meter.record_value.assert_called_once()
   ```

## Best Practices

1. **Descriptive Names**: Use clear names explaining what the fixture provides
2. **Documentation**: Add docstrings to fixtures explaining their purpose
3. **Scope**: Use appropriate scope to balance performance and isolation
4. **Cleanup**: Use yield to clean up resources after tests
5. **Reusability**: Make fixtures generic and composable
6. **Consistency**: Keep fixture setup patterns consistent across tests

## Resources

- [pytest Fixtures Documentation](https://docs.pytest.org/en/latest/how-to/fixtures.html)
- [Fixture Factories Pattern](https://docs.pytest.org/en/latest/how-to/fixtures.html#factories-as-fixtures)
- [Conftest.py in pytest](https://docs.pytest.org/en/latest/reference.html#conftest-py)
