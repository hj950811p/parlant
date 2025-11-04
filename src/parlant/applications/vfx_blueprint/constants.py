# Copyright 2025 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TypeAlias

# Layer Size Constraints (width x height)
LAYER_A_SIZE: TypeAlias = tuple[int, int]
LAYER_A_DIMENSIONS: TypeAlias = tuple[int, int]
LAYER_B_DIMENSIONS: TypeAlias = tuple[int, int]
LAYER_C_DIMENSIONS: TypeAlias = tuple[int, int]
LAYER_D_DIMENSIONS: TypeAlias = tuple[int, int]
LAYER_T_DIMENSIONS: TypeAlias = tuple[int, int]

# Expected dimensions (width, height)
LAYER_DIMENSIONS: dict[str, tuple[int, int]] = {
    "A": (1080, 1920),  # Full-screen background
    "B": (700, 700),    # Overlay element
    "C": (700, 700),    # Overlay element
    "D": (700, 700),    # Overlay element
    "T": (800, 150),    # Title/text overlay
}

# Required fields in blueprint
REQUIRED_LAYERS = {"A", "B", "C", "D", "T"}

# Validation error codes
class ValidationCode:
    """Error codes for blueprint validation"""

    MISSING_REQUIRED_LAYER = "MISSING_REQUIRED_LAYER"
    INVALID_LAYER_DIMENSIONS = "INVALID_LAYER_DIMENSIONS"
    GREEN_SCREEN_CONFLICT = "GREEN_SCREEN_CONFLICT"
    IMPROPER_BACKGROUND_CONTENT = "IMPROPER_BACKGROUND_CONTENT"
    BRACKET_WEIGHT_DETECTED = "BRACKET_WEIGHT_DETECTED"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_FIELD_TYPE = "INVALID_FIELD_TYPE"
    EMPTY_PROMPT = "EMPTY_PROMPT"
    INVALID_STRUCTURE = "INVALID_STRUCTURE"


# Auto-fix codes
class AutoFixCode:
    """Codes for auto-fixes applied"""

    CORRECTED_DIMENSIONS = "CORRECTED_DIMENSIONS"
    REMOVED_BRACKET_WEIGHTS = "REMOVED_BRACKET_WEIGHTS"
    SIMPLIFIED_BACKGROUND_PROMPT = "SIMPLIFIED_BACKGROUND_PROMPT"
    INSERTED_DEFAULT_VALUE = "INSERTED_DEFAULT_VALUE"
    RESOLVED_GREEN_SCREEN = "RESOLVED_GREEN_SCREEN"
    NORMALIZED_PROMPT = "NORMALIZED_PROMPT"
