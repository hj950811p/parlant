# Blueprint Validation and Auto-Fix System

A robust, typed validation and auto-repair system for VFX blueprint JSONs with detailed reporting and logging capabilities.

## Overview

This module provides a comprehensive blueprint validation and auto-fixing system that:

1. **Validates** blueprint structures against strict layer sizing rules and content constraints
2. **Auto-fixes** common violations when possible
3. **Reports** detailed information about errors and fixes with descriptive codes
4. **Enforces** pipeline safety by aborting on unrecoverable errors

## Architecture

### Core Components

- **models.py**: Pydantic-inspired dataclasses for blueprint structures
  - `LayerContent`: Individual layer with dimensions and prompt
  - `Blueprint`: Complete blueprint with all required layers
  - `ValidationError`: Error with code, message, and recoverability flag
  - `AppliedFix`: Applied fix with details
  - `ValidationReport`: Complete report with errors and fixes

- **constants.py**: Layer size constraints and validation/auto-fix codes
  - Layer dimensions: A(1080x1920), B/C/D(700x700), T(800x150)
  - ValidationCode and AutoFixCode constants for structured reporting

- **validation.py**: BlueprintValidator
  - Checks for required layers
  - Validates layer dimensions
  - Detects green-screen conflicts
  - Identifies improper background content
  - Detects bracket-based prompt weights
  - Validates all required fields

- **auto_fix.py**: BlueprintAutoFixer
  - Corrects invalid dimensions
  - Removes bracket weights from prompts
  - Simplifies overly specific background prompts
  - Inserts default prompts for empty fields
  - Resolves green-screen conflicts
  - Maintains immutable copies for rollback support

- **reporter.py**: BlueprintReporter
  - Generates dictionary representation of reports
  - Generates JSON reports
  - Generates human-readable text reports
  - Supports logging integration

- **orchestrator.py**: BlueprintOrchestrator
  - Coordinates validation and auto-fix workflow
  - Provides high-level API for validation + auto-fix
  - Returns structured reports with all details

## Usage

### Simple Validation

```python
from parlant.applications.vfx_blueprint import Blueprint, LayerContent, BlueprintValidator

blueprint = Blueprint(
    layers={
        "A": LayerContent("A", 1080, 1920, "Background"),
        "B": LayerContent("B", 700, 700, "Overlay 1"),
        "C": LayerContent("C", 700, 700, "Overlay 2"),
        "D": LayerContent("D", 700, 700, "Overlay 3"),
        "T": LayerContent("T", 800, 150, "Title"),
    }
)

validator = BlueprintValidator()
report = validator.validate(blueprint)

if report.is_valid:
    print("Blueprint is valid!")
else:
    print(f"Errors found: {len(report.detected_errors)}")
```

### Validation with Auto-Fix

```python
from parlant.applications.vfx_blueprint import BlueprintOrchestrator

orchestrator = BlueprintOrchestrator()
fixed_blueprint, report = orchestrator.validate_and_fix(blueprint, auto_fix=True)

if fixed_blueprint:
    print("Blueprint fixed successfully")
else:
    print("Blueprint has unrecoverable errors")
```

### Generating Reports

```python
# Dictionary report (for APIs/logging)
report_dict = orchestrator.generate_report_dict(report)

# JSON report
json_report = orchestrator.generate_report_json(report)

# Human-readable text
text_report = orchestrator.generate_report_text(report)
print(text_report)
```

## Validation Rules

### Required Layers
All blueprints must include layers A, B, C, D, and T.

### Size Constraints
- Layer A (background): 1080 x 1920
- Layer B (overlay): 700 x 700
- Layer C (overlay): 700 x 700
- Layer D (overlay): 700 x 700
- Layer T (title): 800 x 150

### Content Constraints
- No green-screen keywords in overlay layers (B, C, D)
- No empty prompts
- No bracket-based weights (e.g., `[0.8]prompt`)

## Auto-Fix Rules

### Recoverable Errors
The following errors are automatically fixed:
- Invalid layer dimensions → Corrected to expected size
- Bracket weights in prompts → Removed
- Green-screen keywords in overlays → Removed
- Concrete objects in background → Simplified to generic prompt
- Empty prompts → Filled with layer-appropriate defaults

### Unrecoverable Errors
These errors halt the pipeline:
- Missing required layers
- Invalid field types (width/height must be positive integers)

## Error Codes

### ValidationCode
- `MISSING_REQUIRED_LAYER`: Required layer not found
- `INVALID_LAYER_DIMENSIONS`: Layer dimensions don't match expected size
- `GREEN_SCREEN_CONFLICT`: Green-screen keywords in wrong layer
- `IMPROPER_BACKGROUND_CONTENT`: Problematic content in background
- `BRACKET_WEIGHT_DETECTED`: Bracket weights in prompt
- `MISSING_REQUIRED_FIELD`: Required field missing
- `INVALID_FIELD_TYPE`: Field has invalid type
- `EMPTY_PROMPT`: Prompt is empty or whitespace-only

### AutoFixCode
- `CORRECTED_DIMENSIONS`: Fixed layer dimensions
- `REMOVED_BRACKET_WEIGHTS`: Removed bracket weights from prompt
- `SIMPLIFIED_BACKGROUND_PROMPT`: Simplified overly specific background
- `INSERTED_DEFAULT_VALUE`: Inserted default prompt
- `RESOLVED_GREEN_SCREEN`: Removed green-screen keywords

## Safety & Immutability

- Original blueprint is never modified (deep copies used)
- Supports rollback through immutable copies
- Clear recovery status in reports
- Distinguishes between recoverable and unrecoverable errors

## Testing

Comprehensive test coverage across:
- Models and data structures
- Validation rules (success and failure cases)
- Auto-fix operations and side effects
- Report generation in all formats
- Orchestrator workflows

Run tests with:
```bash
poetry run pytest tests/applications/vfx_blueprint/ -v
```

## Integration

The validation system is designed to integrate into the VFX pipeline's Stage 4 verification:

```python
async def verify_stage(blueprint: Blueprint) -> Blueprint:
    orchestrator = BlueprintOrchestrator()
    fixed, report = orchestrator.validate_and_fix(blueprint, auto_fix=True)
    
    if not fixed:
        raise ValidationError(f"Blueprint validation failed: {report.summary}")
    
    return fixed
```
