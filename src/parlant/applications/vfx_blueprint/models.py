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

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LayerContent:
    """Represents content within a blueprint layer"""

    layer_id: str
    width: int
    height: int
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Blueprint:
    """Represents a complete blueprint structure with all required layers"""

    layers: dict[str, LayerContent]
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"

    def __getitem__(self, key: str) -> LayerContent:
        """Allow dict-like access to layers"""
        return self.layers[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get a layer, with optional default"""
        return self.layers.get(key, default)

    def __contains__(self, key: str) -> bool:
        """Check if layer exists"""
        return key in self.layers


@dataclass
class ValidationError:
    """Represents a validation error with code and message"""

    code: str
    message: str
    layer_id: Optional[str] = None
    recoverable: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary"""
        return {
            "code": self.code,
            "message": self.message,
            "layer_id": self.layer_id,
            "recoverable": self.recoverable,
        }


@dataclass
class AppliedFix:
    """Represents a fix that was applied during auto-fix"""

    code: str
    message: str
    layer_id: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert fix to dictionary"""
        return {
            "code": self.code,
            "message": self.message,
            "layer_id": self.layer_id,
            "details": self.details,
        }


@dataclass
class ValidationReport:
    """Complete validation report with errors and applied fixes"""

    detected_errors: list[ValidationError] = field(default_factory=list)
    applied_fixes: list[AppliedFix] = field(default_factory=list)
    unresolved_errors: list[ValidationError] = field(default_factory=list)
    is_valid: bool = True
    summary: str = ""

    @property
    def has_unrecoverable_errors(self) -> bool:
        """Check if there are unrecoverable errors"""
        return any(not err.recoverable for err in self.unresolved_errors)

    @property
    def recovery_possible(self) -> bool:
        """Check if blueprint can be recovered"""
        return not self.has_unrecoverable_errors

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary"""
        return {
            "is_valid": self.is_valid,
            "summary": self.summary,
            "detected_errors": [err.to_dict() for err in self.detected_errors],
            "applied_fixes": [fix.to_dict() for fix in self.applied_fixes],
            "unresolved_errors": [err.to_dict() for err in self.unresolved_errors],
            "has_unrecoverable_errors": self.has_unrecoverable_errors,
        }
