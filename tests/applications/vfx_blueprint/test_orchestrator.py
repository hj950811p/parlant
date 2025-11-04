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

import pytest

from parlant.applications.vfx_blueprint.models import Blueprint, LayerContent
from parlant.applications.vfx_blueprint.orchestrator import BlueprintOrchestrator


@pytest.fixture
def orchestrator() -> BlueprintOrchestrator:
    """Provide a BlueprintOrchestrator instance"""
    return BlueprintOrchestrator()


@pytest.fixture
def valid_blueprint() -> Blueprint:
    """Provide a valid blueprint"""
    return Blueprint(
        layers={
            "A": LayerContent("A", 1080, 1920, "Natural background"),
            "B": LayerContent("B", 700, 700, "Overlay 1"),
            "C": LayerContent("C", 700, 700, "Overlay 2"),
            "D": LayerContent("D", 700, 700, "Overlay 3"),
            "T": LayerContent("T", 800, 150, "Title"),
        }
    )


class TestBlueprintOrchestratorValidation:
    """Tests for orchestrator validation"""

    def test_that_valid_blueprint_passes_orchestrator(
        self, orchestrator: BlueprintOrchestrator, valid_blueprint: Blueprint
    ) -> None:
        """Verify valid blueprint passes through orchestrator"""
        fixed_blueprint, report = orchestrator.validate_and_fix(valid_blueprint)

        assert report.is_valid is True
        assert fixed_blueprint is not None

    def test_that_orchestrator_returns_original_blueprint_when_valid(
        self, orchestrator: BlueprintOrchestrator, valid_blueprint: Blueprint
    ) -> None:
        """Verify orchestrator returns original valid blueprint"""
        fixed_blueprint, _ = orchestrator.validate_and_fix(valid_blueprint)

        assert fixed_blueprint is valid_blueprint


class TestBlueprintOrchestratorAutoFixing:
    """Tests for orchestrator auto-fix workflow"""

    def test_that_orchestrator_auto_fixes_recoverable_errors(
        self, orchestrator: BlueprintOrchestrator
    ) -> None:
        """Verify orchestrator auto-fixes recoverable errors"""
        blueprint = Blueprint(
            layers={
                "A": LayerContent("A", 640, 480, "Background"),
                "B": LayerContent("B", 700, 700, "Overlay"),
                "C": LayerContent("C", 700, 700, "Overlay"),
                "D": LayerContent("D", 700, 700, "Overlay"),
                "T": LayerContent("T", 800, 150, "Title"),
            }
        )

        fixed_blueprint, report = orchestrator.validate_and_fix(
            blueprint, auto_fix=True
        )

        assert fixed_blueprint is not None
        assert fixed_blueprint.layers["A"].width == 1080
        assert fixed_blueprint.layers["A"].height == 1920

    def test_that_orchestrator_rejects_unrecoverable_errors(
        self, orchestrator: BlueprintOrchestrator
    ) -> None:
        """Verify orchestrator returns None for unrecoverable errors"""
        blueprint = Blueprint(
            layers={
                "B": LayerContent("B", 700, 700, "Overlay"),
            }
        )

        fixed_blueprint, report = orchestrator.validate_and_fix(
            blueprint, auto_fix=True
        )

        assert fixed_blueprint is None
        assert not report.recovery_possible

    def test_that_orchestrator_can_skip_auto_fix(
        self, orchestrator: BlueprintOrchestrator
    ) -> None:
        """Verify orchestrator can skip auto-fix when disabled"""
        blueprint = Blueprint(
            layers={
                "A": LayerContent("A", 640, 480, "Background"),
                "B": LayerContent("B", 700, 700, "Overlay"),
                "C": LayerContent("C", 700, 700, "Overlay"),
                "D": LayerContent("D", 700, 700, "Overlay"),
                "T": LayerContent("T", 800, 150, "Title"),
            }
        )

        fixed_blueprint, report = orchestrator.validate_and_fix(
            blueprint, auto_fix=False
        )

        # Should return the original (not fixed) blueprint
        assert fixed_blueprint is blueprint
        assert not report.is_valid


class TestBlueprintOrchestratorReporting:
    """Tests for orchestrator reporting"""

    def test_that_orchestrator_can_generate_dict_report(
        self, orchestrator: BlueprintOrchestrator, valid_blueprint: Blueprint
    ) -> None:
        """Verify orchestrator can generate dict report"""
        _, report = orchestrator.validate_and_fix(valid_blueprint)

        report_dict = orchestrator.generate_report_dict(report)

        assert "summary" in report_dict
        assert "is_valid" in report_dict

    def test_that_orchestrator_can_generate_json_report(
        self, orchestrator: BlueprintOrchestrator, valid_blueprint: Blueprint
    ) -> None:
        """Verify orchestrator can generate JSON report"""
        _, report = orchestrator.validate_and_fix(valid_blueprint)

        json_report = orchestrator.generate_report_json(report)

        assert isinstance(json_report, str)
        assert "summary" in json_report

    def test_that_orchestrator_can_generate_text_report(
        self, orchestrator: BlueprintOrchestrator, valid_blueprint: Blueprint
    ) -> None:
        """Verify orchestrator can generate text report"""
        _, report = orchestrator.validate_and_fix(valid_blueprint)

        text_report = orchestrator.generate_report_text(report)

        assert "BLUEPRINT VALIDATION REPORT" in text_report
