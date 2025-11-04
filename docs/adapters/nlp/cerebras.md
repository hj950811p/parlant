# Cerebras NLP Service Adapter

## Overview

The Cerebras adapter provides integration with [Cerebras](https://www.cerebras.net/) AI models through the Parlant NLP service interface. It enables high-performance inference using Cerebras' optimized Llama model variants.

## Supported Models

Currently, the following Cerebras models are supported:

- **Llama3.1 8B** (`llama3.1-8b`) - Lightweight model suitable for resource-constrained environments
  - Max tokens: 8,192
  - Recommended for local development and testing
  
- **Llama3.3 70B** (`llama3.3-70b`) - Large model for complex reasoning and generation tasks
  - Max tokens: 32,768 (131,072 context window)
  - Recommended for production deployments requiring sophisticated reasoning

## Setup

### 1. Install Cerebras SDK

Install the Parlant framework with Cerebras support:

```bash
pip install parlant[cerebras]
```

This installs:
- `cerebras-cloud-sdk` - Official Cerebras Python client
- `torch` - Required dependency
- `transformers` - NLP utilities

### 2. Configure API Key

Set your Cerebras API key in the environment:

```bash
export CEREBRAS_API_KEY="your-api-key-here"
```

Verify your setup:

```python
from parlant.adapters.nlp.cerebras_service import CerebrasService

error = CerebrasService.verify_environment()
if error:
    print(f"Setup error: {error}")
else:
    print("Cerebras service is properly configured")
```

## Usage

### Basic Example

```python
import parlant.sdk as p
from parlant.adapters.nlp.cerebras_service import CerebrasService

async def main():
    # Initialize logger and meter
    logger = p.Logger()
    meter = p.Meter()
    
    # Create Cerebras service
    nlp_service = CerebrasService(logger, meter)
    
    # Create an agent using Cerebras
    async with p.Server(nlp_service=nlp_service) as server:
        agent = await server.create_agent(
            name="CerebrasAgent",
            description="Agent powered by Cerebras"
        )
        
        # Use the agent
        response = await agent.chat("Hello!")
```

### Using Different Models

```python
from parlant.adapters.nlp.cerebras_service import (
    Llama3_3_8B,
    Llama3_3_70B,
    CustomCerebrasSchematicGenerator,
)
from parlant.core.loggers import Logger
from parlant.core.meter import Meter

logger = Logger()
meter = Meter()

# Use 8B model
model_8b = Llama3_3_8B(logger, meter)

# Use 70B model
model_70b = Llama3_3_70B(logger, meter)

# Use custom model
from pydantic import BaseModel

class MySchema(BaseModel):
    name: str
    value: int

custom_model = CustomCerebrasSchematicGenerator(
    model_name="llama3.3-70b",
    logger=logger,
    meter=meter,
)
```

### Generation with Temperature Control

```python
# Set temperature for generation
result = await generator.do_generate(
    prompt="Generate a creative response",
    hints={"temperature": 0.8}  # Higher = more creative
)

print(result.content)
print(f"Tokens used: {result.info.usage.input_tokens} input, {result.info.usage.output_tokens} output")
```

## Configuration

### Supported Hints

The Cerebras adapter supports the following generation hints:

| Hint | Type | Default | Description |
|------|------|---------|-------------|
| `temperature` | float | 0.7 | Controls randomness (0.0 = deterministic, 1.0+ = creative) |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `CEREBRAS_API_KEY` | Yes | Your Cerebras API key for authentication |

## Error Handling

The adapter handles several types of errors:

### RateLimitError

Occurs when API rate limits are exceeded.

```python
try:
    result = await generator.do_generate(prompt)
except RateLimitError as e:
    logger.error(f"Rate limit exceeded: {e}")
    # Implement exponential backoff retry logic
    await asyncio.sleep(5)
```

### APIConnectionError

Occurs when unable to connect to Cerebras API.

```python
try:
    result = await generator.do_generate(prompt)
except APIConnectionError as e:
    logger.error(f"Connection failed: {e}")
    # Check network connectivity
```

### APITimeoutError

Occurs when request times out.

```python
try:
    result = await generator.do_generate(prompt)
except APITimeoutError as e:
    logger.error(f"Request timed out: {e}")
    # Consider using a larger timeout
```

### ValidationError

Occurs when the generated response doesn't match the expected schema.

```python
try:
    result = await generator.do_generate(prompt)
except ValidationError as e:
    logger.error(f"Generated content doesn't match schema: {e}")
    # Refine the prompt for better schema adherence
```

## Performance Tuning

### Token Estimation

The adapter uses a tokenizer to estimate token counts before sending requests:

```python
tokenizer = generator.tokenizer
token_count = await tokenizer.estimate_token_count(prompt)
print(f"Estimated tokens: {token_count}")
```

### Monitoring Metrics

Monitor usage and performance:

```python
# After generation
info = result.info
print(f"Model: {info.model}")
print(f"Duration: {info.duration}s")
print(f"Input tokens: {info.usage.input_tokens}")
print(f"Output tokens: {info.usage.output_tokens}")
```

## Troubleshooting

### "CEREBRAS_API_KEY is not set"

**Problem**: API key not configured
**Solution**: Set the environment variable:
```bash
export CEREBRAS_API_KEY="your-key"
```

### "Rate limit exceeded"

**Problem**: Too many requests to Cerebras API
**Solution**: 
- Implement exponential backoff retry logic
- Check your Cerebras account tier and limits
- Reduce request frequency

### "Connection failed"

**Problem**: Unable to connect to Cerebras service
**Solution**:
- Verify internet connectivity
- Check if Cerebras API is operational
- Check firewall/proxy settings

### "Generated content doesn't match schema"

**Problem**: Model generated invalid JSON or structure
**Solution**:
- Use more specific prompts
- Use the 70B model for complex schemas
- Add examples in system prompts
- Lower temperature for more predictable outputs

## Best Practices

1. **Error Handling**: Always wrap API calls in try-except blocks
2. **Token Management**: Check token counts before generation to avoid truncation
3. **Caching**: Cache embeddings and frequently generated responses
4. **Monitoring**: Log and monitor token usage and latencies
5. **Model Selection**: Use 8B for simple tasks, 70B for complex reasoning
6. **Temperature**: Use low temperature (0.3-0.5) for deterministic outputs, higher (0.7-1.0) for creative

## API Reference

See the main [NLP Service documentation](./index.md) for the complete NLP service interface.

For Cerebras API documentation, visit: https://www.cerebras.net/products/cloud/

## Examples

See the `examples/` directory for complete working examples using the Cerebras adapter.
