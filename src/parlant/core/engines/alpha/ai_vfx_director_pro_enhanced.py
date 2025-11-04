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

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from io import StringIO
import json
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from parlant.core.engines.alpha.prompt_builder import PromptBuilder, BuiltInSection, SectionStatus
from parlant.core.common import DefaultBaseModel


class GenerationPhase(Enum):
    """Represents the different phases of the layered generation pipeline."""

    STRUCTURAL_OUTLINE = auto()
    """Phase 1: Generate high-level outline/skeleton of blueprint."""

    DETAILED_FILLS = auto()
    """Phase 2: Generate detailed content conditioned on outline."""

    REFINEMENT = auto()
    """Phase 3: Optional refinement and verification pass."""


class BlueprintSchema(DefaultBaseModel):
    """Pydantic model for VFX blueprint output schema."""

    name: str = Field(description="Name of the blueprint")
    background: str = Field(description="Background context for the blueprint")
    dimensions: List[Dict[str, str]] = Field(
        description="List of dimension specifications with name and value pairs"
    )
    details: Optional[str] = Field(
        default=None,
        description="Additional details about the blueprint"
    )
    compliance_notes: Optional[str] = Field(
        default=None,
        description="Notes about compliance with guardrails"
    )


@dataclass(frozen=True)
class LayeredGenerationResult:
    """Result of the layered generation pipeline."""

    phase1_outline: Dict[str, Any]
    """Output from Phase 1: Structural outline."""

    phase2_fills: Optional[Dict[str, Any]] = None
    """Output from Phase 2: Detailed fills (optional if Phase 1 fails)."""

    phase3_refinement: Optional[Dict[str, Any]] = None
    """Output from Phase 3: Refinement (optional)."""

    completed_phases: List[GenerationPhase] = field(default_factory=list)
    """List of phases that completed successfully."""


class PromptSectionDefinition:
    """Defines modular prompt sections for rules, context, and instructions."""

    @staticmethod
    def rule_definitions() -> str:
        """Build the rules and guardrails section."""
        return """
### BUSINESS RULES AND GUARDRAILS ###

These guardrails must be followed in all generated blueprints:

1. **Dimension Requirements**: 
   - Every blueprint must include at least one dimension specification
   - Each dimension must have a concrete name and value
   - Dimensions should be specific and measurable when possible

2. **Background Requirements**:
   - The background field must be provided for all blueprints
   - Background should provide context for understanding the blueprint
   - Background should reference relevant domain concepts (e.g., culture, setting, genre)

3. **Compliance Standards**:
   - All generated blueprints must follow the BlueprintSchema structure
   - Required fields: name, background, dimensions
   - Optional fields: details, compliance_notes
   - Dimensions must be non-empty list of objects with 'name' and 'value' keys

4. **Quality Standards**:
   - Blueprint names should be descriptive and concise
   - Dimension values should be concrete rather than abstract
   - Background descriptions should be substantive (at least a few words)

5. **Fallback Behavior**:
   - If any refinement step fails, the system will gracefully fall back to the previous phase output
   - The final output must always match the BlueprintSchema
   - Backward compatibility is required for all outputs
"""

    @staticmethod
    def context_template(
        agent_name: str = "VFX Blueprint Director",
        agent_role: str = "Specialized AI for generating VFX blueprints"
    ) -> str:
        """Build the context injection section."""
        return f"""
### CONTEXT AND IDENTITY ###

You are {agent_name}, a specialized AI assistant with the role: {agent_role}.

Your responsibilities:
- Generate high-quality VFX blueprints that meet all guardrails
- Understand the relationship between dimensions and backgrounds
- Ensure all outputs comply with the specified schema
- Provide clear compliance notes when applicable

Current Task Context:
- You are part of a multi-phase generation pipeline
- Earlier phases may have provided outlines or partial specifications
- Your role is to contribute to the iterative refinement of the blueprint
"""

    @staticmethod
    def phase_instructions(phase: GenerationPhase) -> str:
        """Build phase-specific instructions."""
        if phase == GenerationPhase.STRUCTURAL_OUTLINE:
            return """
### PHASE 1: STRUCTURAL OUTLINE ###

Your task is to generate a high-level outline/skeleton for a VFX blueprint.

Instructions:
1. Focus on the core structure: name, background category, and primary dimensions
2. Ensure the background provides meaningful context
3. Include at least 2-3 key dimensions
4. Keep values concrete and specific (not abstract)
5. The outline should be comprehensive enough to guide Phase 2 detailing
6. Output a valid JSON blueprint that follows the BlueprintSchema

Expected output: A complete but concise VFX blueprint JSON object
"""

        elif phase == GenerationPhase.DETAILED_FILLS:
            return """
### PHASE 2: DETAILED FILLS ###

Your task is to enhance and expand the outline from Phase 1 with detailed content.

Instructions:
1. Reference and build upon the outline provided from Phase 1
2. Add more specific details to each dimension
3. Enhance the background with more contextual information
4. Add additional dimensions if they meaningfully enhance the blueprint
5. Populate optional fields (details, compliance_notes) where relevant
6. Maintain consistency with the Phase 1 outline
7. Ensure the output still conforms to BlueprintSchema

Expected output: An enriched and detailed VFX blueprint JSON object
"""

        else:  # REFINEMENT
            return """
### PHASE 3: REFINEMENT AND VERIFICATION ###

Your task is to verify and optionally refine the blueprint from Phase 2.

Instructions:
1. Review the current blueprint against all guardrails
2. Verify all required fields are present and valid
3. Check that dimensions are concrete and measurable
4. Ensure background provides sufficient context
5. Apply any suggested autofixes to improve compliance
6. Add or update compliance_notes to explain any deviations
7. Make minimal changes unless major issues are found

Expected output: A verified and potentially refined VFX blueprint JSON object
"""


class FewShotExemplars:
    """Embedded few-shot exemplars demonstrating compliant VFX blueprints."""

    @staticmethod
    def exemplar_1_basic() -> Dict[str, Any]:
        """Exemplar 1: Basic compliant blueprint (minimal but complete)."""
        return {
            "name": "Simple Sunset Scene",
            "background": "Outdoor landscape during golden hour",
            "dimensions": [
                {"name": "lighting", "value": "warm golden light"},
                {"name": "atmosphere", "value": "peaceful and serene"},
                {"name": "color_palette", "value": "orange, pink, purple tones"}
            ],
            "details": "A simple outdoor scene with natural lighting and peaceful ambiance",
            "compliance_notes": "Meets all basic guardrails: concrete dimensions, meaningful background"
        }

    @staticmethod
    def exemplar_2_complex() -> Dict[str, Any]:
        """Exemplar 2: Complex compliant blueprint with multiple dimensions."""
        return {
            "name": "High-Tech Space Station Interior",
            "background": "Sci-fi setting featuring advanced technology in a space environment",
            "dimensions": [
                {"name": "technology_level", "value": "advanced futuristic"},
                {"name": "primary_colors", "value": "metallic blues and silvers"},
                {"name": "lighting_style", "value": "neon accents with cool LED strips"},
                {"name": "environment_type", "value": "enclosed space station corridor"},
                {"name": "atmosphere_mood", "value": "sterile but professional"},
                {"name": "material_appearance", "value": "polished metal, glass, and composite surfaces"}
            ],
            "details": "A complex sci-fi environment featuring multiple technical elements, advanced lighting design, and carefully chosen materials to convey a cutting-edge atmosphere",
            "compliance_notes": "Comprehensive blueprint with 6 dimensions, all concrete and measurable. Background provides rich sci-fi context. Suitable for detailed VFX work."
        }

    @staticmethod
    def exemplar_3_guardrails() -> Dict[str, Any]:
        """Exemplar 3: Demonstrates guardrail compliance and edge-case handling."""
        return {
            "name": "Ancient Temple with Modern Overlay",
            "background": "Historical Asian temple architecture blended with contemporary digital elements, representing past-meets-future theme",
            "dimensions": [
                {"name": "architectural_style", "value": "classical Asian temple with stone and wood"},
                {"name": "period_setting", "value": "historical base with modern tech enhancements"},
                {"name": "lighting_treatment", "value": "natural sunlight through ancient stone combined with digital blue glows"},
                {"name": "cultural_references", "value": "Buddhist and Hindu iconography with digital holographic elements"},
                {"name": "visual_hierarchy", "value": "stone and traditional materials in foreground, digital overlays in mid-ground"}
            ],
            "details": "This blueprint exemplifies how to handle culturally-sensitive content while maintaining guardrails. Background clearly references culture, all dimensions are concrete, and the design respects both traditional and modern visual languages.",
            "compliance_notes": "Excellent compliance: culturally-aware background, concrete dimensions with specific visual references, appropriate detail level for complex multi-disciplinary VFX work"
        }

    @staticmethod
    def get_all() -> List[Dict[str, Any]]:
        """Get all exemplars as list of dictionaries."""
        return [
            FewShotExemplars.exemplar_1_basic(),
            FewShotExemplars.exemplar_2_complex(),
            FewShotExemplars.exemplar_3_guardrails()
        ]

    @staticmethod
    def get_all_with_metadata() -> List[Dict[str, Any]]:
        """Get all exemplars with metadata about their purpose."""
        return [
            {
                "blueprint": FewShotExemplars.exemplar_1_basic(),
                "description": "Basic compliant blueprint demonstrating minimal but complete structure",
                "compliance_notes": "Shows baseline guardrail compliance"
            },
            {
                "blueprint": FewShotExemplars.exemplar_2_complex(),
                "description": "Complex blueprint with multiple dimensions and detailed specifications",
                "compliance_notes": "Demonstrates comprehensive dimension specification and complex scene description"
            },
            {
                "blueprint": FewShotExemplars.exemplar_3_guardrails(),
                "description": "Edge-case blueprint showing cultural sensitivity and complex guardrail handling",
                "compliance_notes": "Shows best practices for culturally-aware and complex multi-element blueprints"
            }
        ]

    @staticmethod
    def format_for_prompt() -> str:
        """Format all exemplars as a readable section for prompts."""
        buffer = StringIO()
        buffer.write("\n### REFERENCE EXEMPLARS ###\n\n")
        buffer.write("The following are exemplars of compliant VFX blueprints:\n\n")

        for i, exemplar_meta in enumerate(FewShotExemplars.get_all_with_metadata(), 1):
            buffer.write(f"**Exemplar {i}: {exemplar_meta['blueprint']['name']}**\n")
            buffer.write(f"Description: {exemplar_meta['description']}\n")
            buffer.write(f"Compliance: {exemplar_meta['compliance_notes']}\n\n")
            buffer.write(f"```json\n{json.dumps(exemplar_meta['blueprint'], indent=2)}\n```\n\n")

        return buffer.getvalue()


class AIVFXDirector:
    """
    Main orchestrator for the layered VFX blueprint generation pipeline.

    Implements a three-phase generation process:
    1. Structural Outline: High-level skeleton of the blueprint
    2. Detailed Fills: Enriched content conditioned on outline
    3. Refinement: Optional verification and improvement
    
    Includes fallback support for backward compatibility and graceful degradation.
    """

    def __init__(self) -> None:
        """Initialize the VFX Director."""
        self._available_sections = {
            "rules": PromptSectionDefinition.rule_definitions,
            "context": PromptSectionDefinition.context_template,
            "instructions": PromptSectionDefinition.phase_instructions,
        }

    def get_available_sections(self) -> Dict[str, Any]:
        """Get list of available prompt sections."""
        return list(self._available_sections.keys())  # type: ignore

    def _build_rules_section(self) -> str:
        """Build the rules and guardrails section."""
        return PromptSectionDefinition.rule_definitions()

    def _build_context_section(
        self,
        agent_name: str = "VFX Blueprint Director",
        agent_role: str = "Specialized AI for generating VFX blueprints"
    ) -> str:
        """Build the context injection section."""
        return PromptSectionDefinition.context_template(agent_name, agent_role)

    def _build_phase_instructions(self, phase: GenerationPhase) -> str:
        """Build phase-specific instructions."""
        return PromptSectionDefinition.phase_instructions(phase)

    def _build_phase_prompt(
        self,
        phase: GenerationPhase,
        user_requirement: Optional[str] = None,
        previous_phase_output: Optional[Dict[str, Any]] = None,
        custom_guardrails: Optional[List[str]] = None,
        autofix_hints: Optional[Dict[str, str]] = None,
        include_exemplars: bool = True
    ) -> str:
        """Build a complete prompt for a given phase."""
        buffer = StringIO()

        buffer.write(self._build_rules_section())
        buffer.write("\n\n")

        if custom_guardrails:
            buffer.write("### CUSTOM GUARDRAILS ###\n\n")
            for guardrail in custom_guardrails:
                buffer.write(f"- {guardrail}\n")
            buffer.write("\n\n")

        buffer.write(self._build_context_section())
        buffer.write("\n\n")

        if include_exemplars and phase == GenerationPhase.STRUCTURAL_OUTLINE:
            buffer.write(FewShotExemplars.format_for_prompt())
            buffer.write("\n\n")

        buffer.write(self._build_phase_instructions(phase))
        buffer.write("\n\n")

        if user_requirement:
            buffer.write("### USER REQUIREMENT ###\n\n")
            buffer.write(f"{user_requirement}\n\n")

        if previous_phase_output:
            buffer.write("### PREVIOUS PHASE OUTPUT (for conditioning) ###\n\n")
            buffer.write("```json\n")
            buffer.write(json.dumps(previous_phase_output, indent=2))
            buffer.write("\n```\n\n")

        if autofix_hints:
            buffer.write("### AUTOFIX HINTS ###\n\n")
            buffer.write("The following issues were detected in the previous output.\n")
            buffer.write("Consider these suggestions when generating your output:\n\n")
            for key, hint in autofix_hints.items():
                buffer.write(f"- {key}: {hint}\n")
            buffer.write("\n")

        buffer.write(
            "Generate the output as a valid JSON object matching the BlueprintSchema structure.\n"
        )

        return buffer.getvalue()

    def create_prompt_builder_for_phase(
        self,
        phase: GenerationPhase,
        user_requirement: Optional[str] = None,
        previous_phase_output: Optional[Dict[str, Any]] = None,
        custom_rules: Optional[List[str]] = None,
        custom_context: Optional[Dict[str, str]] = None
    ) -> PromptBuilder:
        """Create a PromptBuilder for a specific phase."""
        builder = PromptBuilder()

        builder.add_section(
            name="rules",
            template=self._build_rules_section(),
            status=SectionStatus.ACTIVE,
        )

        if custom_rules:
            custom_rules_text = "\n".join(f"- {rule}" for rule in custom_rules)
            builder.add_section(
                name="custom_rules",
                template=f"### CUSTOM RULES ###\n\n{custom_rules_text}",
                status=SectionStatus.ACTIVE,
            )

        agent_name = custom_context.get("agent_name", "VFX Blueprint Director") if custom_context else "VFX Blueprint Director"
        agent_role = custom_context.get("agent_role", "Specialized AI for generating VFX blueprints") if custom_context else "Specialized AI for generating VFX blueprints"

        builder.add_section(
            name="context",
            template=self._build_context_section(agent_name, agent_role),
            status=SectionStatus.ACTIVE,
        )

        if phase == GenerationPhase.STRUCTURAL_OUTLINE:
            builder.add_section(
                name="exemplars",
                template=FewShotExemplars.format_for_prompt(),
                status=SectionStatus.ACTIVE,
            )

        builder.add_section(
            name="instructions",
            template=self._build_phase_instructions(phase),
            status=SectionStatus.ACTIVE,
        )

        if user_requirement:
            builder.add_section(
                name="user_requirement",
                template=f"### USER REQUIREMENT ###\n\n{user_requirement}",
                status=SectionStatus.ACTIVE,
            )

        if previous_phase_output:
            output_json = json.dumps(previous_phase_output, indent=2)
            builder.add_section(
                name="previous_output",
                template=f"### PREVIOUS PHASE OUTPUT ###\n\n```json\n{output_json}\n```",
                status=SectionStatus.ACTIVE,
            )

        return builder

    def get_final_output(self, result: LayeredGenerationResult) -> Dict[str, Any]:
        """
        Extract the final output from a layered generation result.

        Implements graceful fallback: uses the latest available phase output.
        """
        if result.phase3_refinement:
            return result.phase3_refinement
        elif result.phase2_fills:
            return result.phase2_fills
        else:
            return result.phase1_outline

    @staticmethod
    def validate_blueprint_schema(output: Dict[str, Any]) -> bool:
        """Validate that output matches BlueprintSchema."""
        try:
            BlueprintSchema.model_validate(output)
            return True
        except Exception:
            return False

    @staticmethod
    def extract_compliance_issues(output: Dict[str, Any]) -> List[str]:
        """Extract compliance issues from a blueprint output."""
        issues: List[str] = []

        if not output.get("name"):
            issues.append("Missing required field: name")

        if not output.get("background"):
            issues.append("Missing required field: background")

        if not output.get("dimensions") or not isinstance(output["dimensions"], list):
            issues.append("Missing or invalid required field: dimensions")
        elif len(output["dimensions"]) == 0:
            issues.append("Dimensions list must not be empty")
        else:
            for i, dim in enumerate(output["dimensions"]):
                if not dim.get("name"):
                    issues.append(f"Dimension {i}: missing 'name'")
                if not dim.get("value"):
                    issues.append(f"Dimension {i}: missing or empty 'value'")

        return issues
