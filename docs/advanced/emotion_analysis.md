# Emotion Analysis with Semantic Understanding

## Overview

Parlant's emotion analysis system uses AI-powered semantic analysis to detect and interpret emotional content in customer messages. This enables more nuanced and contextually appropriate agent responses, with timing and intensity adjustments for animation and effects.

## Architecture

The emotion analysis system follows Parlant's Hexagonal Architecture:

- **Core Domain** (`src/parlant/core/nlp/emotion.py`): Defines the `EmotionAnalyzer` interface and `EmotionAnalysis` dataclass
- **Adapters** (`src/parlant/adapters/nlp/emotion_analyzer.py`): Implements Cerebras-based semantic analysis with intelligent fallback
- **Service Integration** (`src/parlant/core/nlp/service.py`): Integrates emotion analysis into the NLP service layer

## EmotionAnalysis Structure

Every emotion analysis returns a structured `EmotionAnalysis` object:

```python
@dataclass(frozen=True)
class EmotionAnalysis:
    intensity: int                          # 1-10 scale
    pacing: Literal["fast", "slow"]        # Recommended delivery pace
    focus_keywords: Sequence[str]          # Key emotional terms
    recommended_scene_duration: int        # Duration in milliseconds
```

### Field Descriptions

- **intensity**: Scale of 1-10 representing the strength of the emotion
  - 1-3: Neutral, informational
  - 4-6: Moderately emotional
  - 7-10: Highly emotional, urgent, or intense

- **pacing**: Recommended delivery pace
  - `"fast"`: For energetic, urgent, or high-energy emotions (happiness, anger, excitement, frustration)
  - `"slow"`: For calm, contemplative, or low-energy emotions (sadness, contentment, confusion)

- **focus_keywords**: 1-3 key words capturing the emotional tone
  - Extracted from the semantic analysis or heuristic fallback
  - Used for animation trigger points and context

- **recommended_scene_duration**: Suggested animation/scene duration in milliseconds
  - Typically 3000-6000ms based on intensity and text length
  - Can be used to synchronize visual effects timing

## Usage

### Accessing the Emotion Analyzer

```python
from lagom import Container
from parlant.core.nlp.service import NLPService

container: Container
nlp_service = container[NLPService]
emotion_analyzer = await nlp_service.get_emotion_analyzer()

result = await emotion_analyzer.analyze("I'm absolutely thrilled about this!")
print(result.intensity)  # 8-9
print(result.pacing)     # "fast"
print(result.focus_keywords)  # ["thrilled"]
print(result.recommended_scene_duration)  # 4500ms
```

### Integration with Message Generation

Emotion analysis can be integrated into message generation to:

1. **Adjust response pacing** in Stage 2 prompt tailoring
2. **Enrich animation metadata** in Stage 3 for visual effects timing
3. **Tag messages** with emotional context for post-processing

## Implementation Details

### Cerebras-Based Analysis

The primary implementation uses Cerebras LLM with structured JSON schema validation:

```python
class CerebrasEmotionAnalyzer(EmotionAnalyzer):
    """Uses Llama 3.3-70B via Cerebras with intelligent fallback."""
```

The analyzer sends a prompt to Cerebras requesting structured emotion analysis and expects:

```json
{
    "intensity": 8,
    "pacing": "fast",
    "focus_keywords": ["frustrated", "terrible"],
    "recommended_scene_duration": 5000
}
```

### Fallback Heuristic Analysis

When Cerebras API is unavailable or fails:

1. **Keyword-based detection**: Counts positive/negative keywords in the text
2. **Intensity calculation**: Based on keyword frequency and text length
3. **Pacing determination**: High emotion keywords trigger "fast" pacing
4. **Focus extraction**: Top emotional keywords are extracted
5. **Duration estimation**: Calculated based on intensity and message length

This ensures graceful degradation without service interruption.

### Caching

Repeated analyses of the same text are cached using MD5 hashing:

```python
analyzer = CerebrasEmotionAnalyzer(logger, meter)
result1 = await analyzer.analyze("Same text")
result2 = await analyzer.analyze("Same text")  # Served from cache
# result1 == result2, with only 1 Cerebras API call
```

## Configuration

### Environment Variables

- `CEREBRAS_API_KEY`: Required for Cerebras-based emotion analysis
  - The adapter will use the existing Cerebras configuration
  - Same authentication/API key as other Cerebras services

### Service Selection

All NLP service providers include the emotion analyzer:

- OpenAI
- Anthropic
- Azure
- Google Vertex
- Cerebras
- Ollama
- And others...

They all use the same Cerebras-backed `CerebrasEmotionAnalyzer` for consistent semantic understanding.

## Best Practices

1. **Cache for performance**: The analyzer caches results automatically
2. **Use focus_keywords for context**: Consider the emotional focus when crafting responses
3. **Respect pacing in animations**: Use recommended_scene_duration for synchronized effects
4. **Monitor fallback usage**: If fallback is frequently used, check Cerebras API status
5. **Combine with other signals**: Use emotion analysis alongside other semantic understanding

## Error Handling

The emotion analyzer handles failures gracefully:

```python
try:
    result = await analyzer.analyze("Customer message")
except Exception as e:
    # Analyzer will attempt fallback heuristic
    # If heuristic also fails, exception is raised
    logger.error(f"Emotion analysis failed: {e}")
```

All exceptions are logged with context, and the system will continue functioning with heuristic-based analysis.

## Integration Examples

### Stage 2: Prompt Tailoring

Adjust agent prompts based on detected emotion:

```python
emotion_result = await emotion_analyzer.analyze(customer_message)
if emotion_result.intensity >= 7:
    # Add empathy to prompt for high-emotion customers
    prompt += "\nPlease demonstrate empathy given the customer's emotional state."
```

### Stage 3: Animation Enrichment

Enrich message events with animation metadata:

```python
emotion_result = await emotion_analyzer.analyze(agent_response)
message_event.metadata = {
    "emotion_pacing": emotion_result.pacing,
    "animation_duration_ms": emotion_result.recommended_scene_duration,
    "focus_keywords": emotion_result.focus_keywords,
}
```

## Testing

Emotion analysis is thoroughly tested:

- **Unit tests** mock Cerebras responses for typical emotions
- **Fallback tests** ensure graceful degradation
- **Caching tests** verify performance optimization
- **Integration tests** validate end-to-end flows

Tests are located in `tests/core/stable/nlp/test_emotion_analyzer.py`.

## Performance Considerations

- **First call**: 50-200ms (Cerebras API + parsing)
- **Cached calls**: <1ms (hash lookup + dictionary retrieval)
- **Fallback**: 1-5ms (keyword matching)
- **Cache size**: Limited by Python's LRU cache (128 entries default)

## Future Enhancements

Potential improvements for future versions:

1. **Emotion history tracking**: Analyze sentiment trends over conversation
2. **Multi-turn emotion dynamics**: Consider emotion changes across exchanges
3. **Cultural context awareness**: Adapt analysis for different cultural contexts
4. **Custom keyword sets**: Allow per-application emotion vocabularies
5. **Real-time confidence scores**: Include LLM confidence in emotion detection
