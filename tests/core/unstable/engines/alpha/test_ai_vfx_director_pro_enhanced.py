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

import json
import pytest
from typing import Any

from parlant.core.engines.alpha.ai_vfx_director_pro_enhanced import (
    BlueprintSchema,
    FewShotExemplars,
    PromptSectionDefinition,
    AIVFXDirector,
    GenerationPhase,
    LayeredGenerationResult,
)
from parlant.core.engines.alpha.prompt_builder import PromptBuilder


class TestBlueprintFewShotExemplars:
    """Test that few-shot exemplars are valid and consistent."""

    def test_blueprint_few_shot_exemplars_are_valid(self) -> None:
        exemplars = FewShotExemplars.get_all()
        assert len(exemplars) >= 2, "Should have at least 2 exemplars"
        
        for i, exemplar in enumerate(exemplars):
            assert isinstance(exemplar, dict), f"Exemplar {i} should be a dict"
            blueprint = BlueprintSchema.model_validate(exemplar)
            assert blueprint.name, f"Exemplar {i} should have a name"
            assert blueprint.dimensions, f"Exemplar {i} should have dimensions"

    def test_exemplar_provenance_is_documented(self) -> None:
        exemplars = FewShotExemplars.get_all_with_metadata()
        assert len(exemplars) >= 2
        
        for exemplar_meta in exemplars:
            assert "blueprint" in exemplar_meta
            assert "description" in exemplar_meta
            assert "compliance_notes" in exemplar_meta
            assert exemplar_meta["description"], "Description should not be empty"
            assert exemplar_meta["compliance_notes"], "Compliance notes should not be empty"

    def test_all_exemplars_comply_with_guardrails(self) -> None:
        exemplars = FewShotExemplars.get_all()
        for exemplar in exemplars:
            blueprint = BlueprintSchema.model_validate(exemplar)
            
            assert blueprint.background, "Blueprint must have background"
            assert len(blueprint.dimensions) > 0, "Blueprint must have at least one dimension"
            for dim in blueprint.dimensions:
                assert dim.get("name"), "Each dimension must have a name"
                assert dim.get("value"), "Each dimension must have a value"


class TestPromptSectionModularity:
    """Test that prompt sections are modular and adjustable."""

    def test_modular_prompt_construction_basic(self) -> None:
        director = AIVFXDirector()
        
        sections = director.get_available_sections()
        assert "rules" in sections
        assert "context" in sections
        assert "instructions" in sections

    def test_rule_definitions_section_contains_guardrails(self) -> None:
        director = AIVFXDirector()
        rules_section = director._build_rules_section()
        
        assert "dimension" in rules_section.lower()
        assert "background" in rules_section.lower()
        assert "guardrail" in rules_section.lower()

    def test_context_injection_section_configurable(self) -> None:
        director = AIVFXDirector()
        
        context_section = director._build_context_section(
            agent_name="Test Agent",
            agent_role="Blueprint Director"
        )
        
        assert "Test Agent" in context_section
        assert "Blueprint Director" in context_section

    def test_instruction_blocks_vary_by_phase(self) -> None:
        director = AIVFXDirector()
        
        outline_instructions = director._build_phase_instructions(GenerationPhase.STRUCTURAL_OUTLINE)
        detailed_instructions = director._build_phase_instructions(GenerationPhase.DETAILED_FILLS)
        refinement_instructions = director._build_phase_instructions(GenerationPhase.REFINEMENT)
        
        assert "outline" in outline_instructions.lower() or "structure" in outline_instructions.lower()
        assert "detail" in detailed_instructions.lower() or "fill" in detailed_instructions.lower()
        assert "refine" in refinement_instructions.lower() or "verify" in refinement_instructions.lower()

    def test_prompt_with_different_guardrails(self) -> None:
        director = AIVFXDirector()
        
        prompt_default = director._build_phase_prompt(
            GenerationPhase.STRUCTURAL_OUTLINE,
            custom_guardrails=None
        )
        
        custom_guardrails = ["Dimension values must be numeric", "Background must reference culture"]
        prompt_custom = director._build_phase_prompt(
            GenerationPhase.STRUCTURAL_OUTLINE,
            custom_guardrails=custom_guardrails
        )
        
        assert len(prompt_custom) > len(prompt_default)
        for guardrail in custom_guardrails:
            assert guardrail in prompt_custom


class TestStructuralOutlineGeneration:
    """Test Phase 1: Structural Outline Generation."""

    @pytest.mark.asyncio
    async def test_structural_outline_generation_creates_valid_outline(self) -> None:
        director = AIVFXDirector()
        
        outline_prompt = director._build_phase_prompt(GenerationPhase.STRUCTURAL_OUTLINE)
        assert outline_prompt
        assert isinstance(outline_prompt, str)
        assert len(outline_prompt) > 0

    def test_outline_prompt_includes_exemplars(self) -> None:
        director = AIVFXDirector()
        
        outline_prompt = director._build_phase_prompt(
            GenerationPhase.STRUCTURAL_OUTLINE,
            include_exemplars=True
        )
        
        assert "exemplar" in outline_prompt.lower() or "example" in outline_prompt.lower()

    def test_outline_prompt_with_user_requirement(self) -> None:
        director = AIVFXDirector()
        user_requirement = "Create a blueprint for a fantasy scene with magical effects"
        
        outline_prompt = director._build_phase_prompt(
            GenerationPhase.STRUCTURAL_OUTLINE,
            user_requirement=user_requirement
        )
        
        assert user_requirement in outline_prompt or "fantasy" in outline_prompt.lower()


class TestDetailedFillsGeneration:
    """Test Phase 2: Detailed Fills Conditioned on Outline."""

    def test_detailed_fills_prompt_references_outline(self) -> None:
        director = AIVFXDirector()
        
        outline_dict = {
            "name": "Test Blueprint",
            "dimensions": [{"name": "lighting", "value": "dramatic"}],
            "background": "outdoor"
        }
        
        fills_prompt = director._build_phase_prompt(
            GenerationPhase.DETAILED_FILLS,
            previous_phase_output=outline_dict
        )
        
        assert fills_prompt
        assert "Test Blueprint" in fills_prompt
        assert "dramatic" in fills_prompt or "lighting" in fills_prompt

    def test_detailed_fills_includes_conditioning_context(self) -> None:
        director = AIVFXDirector()
        
        outline_dict = {
            "name": "Sci-Fi Blueprint",
            "dimensions": [{"name": "technology", "value": "futuristic"}],
            "background": "space station"
        }
        
        fills_prompt = director._build_phase_prompt(
            GenerationPhase.DETAILED_FILLS,
            previous_phase_output=outline_dict
        )
        
        assert "Sci-Fi Blueprint" in fills_prompt
        assert "conditioned" in fills_prompt.lower() or "based on" in fills_prompt.lower()


class TestRefinementPhase:
    """Test Phase 3: Optional Refinement/Verification."""

    def test_refinement_prompt_includes_validation_criteria(self) -> None:
        director = AIVFXDirector()
        
        blueprint_dict = {
            "name": "Draft Blueprint",
            "dimensions": [{"name": "color", "value": "blue"}],
            "background": "ocean"
        }
        
        refinement_prompt = director._build_phase_prompt(
            GenerationPhase.REFINEMENT,
            previous_phase_output=blueprint_dict
        )
        
        assert refinement_prompt
        assert "verify" in refinement_prompt.lower() or "validate" in refinement_prompt.lower() or "refine" in refinement_prompt.lower()

    def test_refinement_phase_can_use_autofix_hints(self) -> None:
        director = AIVFXDirector()
        
        blueprint_dict = {
            "name": "Blueprint with Issues",
            "dimensions": [{"name": "incomplete", "value": ""}],
            "background": ""
        }
        
        autofix_hints = {
            "missing_background": "Infer from context",
            "empty_dimension_value": "Suggest reasonable value"
        }
        
        refinement_prompt = director._build_phase_prompt(
            GenerationPhase.REFINEMENT,
            previous_phase_output=blueprint_dict,
            autofix_hints=autofix_hints
        )
        
        assert "autofix" in refinement_prompt.lower() or "fix" in refinement_prompt.lower() or "suggest" in refinement_prompt.lower()


class TestFallbackAndBackwardCompatibility:
    """Test fallback mechanisms and backward compatibility."""

    @pytest.mark.asyncio
    async def test_fallback_when_refinement_fails(self) -> None:
        director = AIVFXDirector()
        
        outline = {
            "name": "Fallback Test Blueprint",
            "dimensions": [{"name": "test", "value": "value"}],
            "background": "test background"
        }
        
        fills = {
            "name": "Fallback Test Blueprint",
            "dimensions": [
                {"name": "test", "value": "value"},
                {"name": "detail", "value": "filled in"}
            ],
            "background": "test background with details"
        }
        
        result = LayeredGenerationResult(
            phase1_outline=outline,
            phase2_fills=fills,
            phase3_refinement=None,
            completed_phases=[GenerationPhase.STRUCTURAL_OUTLINE, GenerationPhase.DETAILED_FILLS]
        )
        
        final_output = director.get_final_output(result)
        assert final_output
        assert BlueprintSchema.model_validate(final_output)

    def test_final_output_matches_original_schema(self) -> None:
        director = AIVFXDirector()
        
        test_output = {
            "name": "Schema Test Blueprint",
            "dimensions": [{"name": "test_dim", "value": "test_val"}],
            "background": "test_bg"
        }
        
        blueprint = BlueprintSchema.model_validate(test_output)
        assert blueprint.name == "Schema Test Blueprint"
        assert len(blueprint.dimensions) == 1
        assert blueprint.background == "test_bg"

    def test_backward_compatibility_single_phase_fallback(self) -> None:
        director = AIVFXDirector()
        
        only_outline = {
            "name": "Only Outline Blueprint",
            "dimensions": [{"name": "basic", "value": "simple"}],
            "background": "basic background"
        }
        
        result = LayeredGenerationResult(
            phase1_outline=only_outline,
            phase2_fills=None,
            phase3_refinement=None,
            completed_phases=[GenerationPhase.STRUCTURAL_OUTLINE]
        )
        
        final_output = director.get_final_output(result)
        blueprint = BlueprintSchema.model_validate(final_output)
        assert blueprint.name == "Only Outline Blueprint"


class TestLayeredGenerationFlow:
    """Test the overall layered generation flow."""

    def test_layered_generation_result_structure(self) -> None:
        outline = {
            "name": "Flow Test",
            "dimensions": [{"name": "d1", "value": "v1"}],
            "background": "bg"
        }
        fills = {
            "name": "Flow Test",
            "dimensions": [
                {"name": "d1", "value": "v1"},
                {"name": "d2", "value": "v2"}
            ],
            "background": "bg with more details"
        }
        
        result = LayeredGenerationResult(
            phase1_outline=outline,
            phase2_fills=fills,
            phase3_refinement=None,
            completed_phases=[GenerationPhase.STRUCTURAL_OUTLINE, GenerationPhase.DETAILED_FILLS]
        )
        
        assert result.phase1_outline == outline
        assert result.phase2_fills == fills
        assert result.phase3_refinement is None
        assert GenerationPhase.STRUCTURAL_OUTLINE in result.completed_phases
        assert GenerationPhase.DETAILED_FILLS in result.completed_phases

    def test_director_prompt_builder_integration(self) -> None:
        director = AIVFXDirector()
        
        builder = director.create_prompt_builder_for_phase(
            GenerationPhase.STRUCTURAL_OUTLINE
        )
        
        assert isinstance(builder, PromptBuilder)
        prompt = builder.build()
        assert prompt
        assert isinstance(prompt, str)

    def test_modular_and_adjustable_prompt_sections(self) -> None:
        director = AIVFXDirector()
        
        custom_rules = [
            "All blueprints must specify color palette",
            "Dimensions must be concrete, not abstract"
        ]
        custom_context = {"agent_name": "VFX Director AI"}
        
        builder = director.create_prompt_builder_for_phase(
            GenerationPhase.STRUCTURAL_OUTLINE,
            custom_rules=custom_rules,
            custom_context=custom_context
        )
        
        prompt = builder.build()
        assert "color palette" in prompt
        assert "concrete" in prompt
        assert "VFX Director AI" in prompt
