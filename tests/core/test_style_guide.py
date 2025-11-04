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

from parlant.core.style_guide import (
    StyleGuide,
    generate_global_style_guide,
    AnimationCadence,
)


class TestStyleGuideGeneration:
    """Tests for style guide generation from dialogues."""

    async def test_that_generate_global_style_guide_creates_style_guide_from_dialogues(
        self,
    ) -> None:
        """Test that a style guide is generated from dialogue analysis."""
        dialogues = [
            "The sunset paints the sky in warm orange and golden hues",
            "Colors dance across the horizon in rapid succession",
            "Fast-paced action with vibrant palettes",
        ]

        style_guide = await generate_global_style_guide(dialogues)

        assert isinstance(style_guide, StyleGuide)
        assert style_guide.palette is not None
        assert style_guide.visual_style is not None
        assert style_guide.animation_cadence is not None
        assert style_guide.mood is not None

    async def test_that_style_guide_includes_palette_visual_style_animation_mood(
        self,
    ) -> None:
        """Test that generated style guide includes all required attributes."""
        dialogues = [
            "Dark blue and purple tones dominate the scene",
            "Minimalist and clean aesthetic throughout",
            "Slow, methodical camera movements",
            "Somber and contemplative atmosphere",
        ]

        style_guide = await generate_global_style_guide(dialogues)

        # Verify all key attributes are present
        assert hasattr(style_guide, "palette")
        assert hasattr(style_guide, "visual_style")
        assert hasattr(style_guide, "animation_cadence")
        assert hasattr(style_guide, "mood")

        # Verify they have meaningful values
        assert isinstance(style_guide.palette, list) or style_guide.palette is not None
        assert isinstance(style_guide.visual_style, str)
        assert isinstance(style_guide.animation_cadence, AnimationCadence)
        assert isinstance(style_guide.mood, str)

    async def test_that_user_overrides_take_precedence_over_generated_guide(
        self,
    ) -> None:
        """Test that user overrides take precedence over generated guide."""
        dialogues = [
            "Warm orange sunset colors",
            "Fast-paced action scenes",
        ]

        user_overrides = {
            "palette": ["#FF0000", "#00FF00", "#0000FF"],
            "visual_style": "cyberpunk",
            "mood": "energetic",
        }

        style_guide = await generate_global_style_guide(dialogues, user_overrides=user_overrides)

        # Verify overrides are applied
        assert style_guide.palette == ["#FF0000", "#00FF00", "#0000FF"]
        assert style_guide.visual_style == "cyberpunk"
        assert style_guide.mood == "energetic"

    async def test_that_partial_user_overrides_merge_with_generated_values(
        self,
    ) -> None:
        """Test that partial overrides merge with generated values."""
        dialogues = [
            "Slow, meditative scenes with cool tones",
            "Peaceful atmosphere throughout",
        ]

        user_overrides = {
            "mood": "peaceful",
        }

        style_guide = await generate_global_style_guide(dialogues, user_overrides=user_overrides)

        # Mood should be overridden
        assert style_guide.mood == "peaceful"
        # Other values should be generated
        assert style_guide.palette is not None
        assert style_guide.visual_style is not None
        assert style_guide.animation_cadence is not None

    async def test_that_style_guide_can_be_serialized_for_metadata(self) -> None:
        """Test that style guide can be serialized to dict for metadata output."""
        dialogues = [
            "Colorful and vibrant scenes",
            "Dynamic and fast-paced action",
        ]

        style_guide = await generate_global_style_guide(dialogues)
        serialized = style_guide.to_dict()

        assert isinstance(serialized, dict)
        assert "palette" in serialized
        assert "visual_style" in serialized
        assert "animation_cadence" in serialized
        assert "mood" in serialized

    async def test_that_empty_dialogues_produce_default_style_guide(self) -> None:
        """Test that empty dialogues produce a valid default style guide."""
        dialogues: list[str] = []

        style_guide = await generate_global_style_guide(dialogues)

        assert isinstance(style_guide, StyleGuide)
        assert style_guide.palette is not None
        assert style_guide.visual_style is not None
        assert style_guide.animation_cadence is not None
        assert style_guide.mood is not None

    async def test_that_style_guide_has_consistent_values_across_calls(self) -> None:
        """Test that style guides derived from same dialogues are consistent."""
        dialogues = [
            "Professional corporate environment",
            "Blue and white color scheme",
            "Steady, professional pacing",
        ]

        style_guide1 = await generate_global_style_guide(dialogues)
        style_guide2 = await generate_global_style_guide(dialogues)

        # Same dialogues should produce same or very similar results
        assert style_guide1.visual_style == style_guide2.visual_style
        assert style_guide1.animation_cadence == style_guide2.animation_cadence


class TestStyleGuideModel:
    """Tests for StyleGuide data model."""

    def test_that_style_guide_can_be_instantiated(self) -> None:
        """Test basic StyleGuide instantiation."""
        style_guide = StyleGuide(
            palette=["#FF0000", "#00FF00"],
            visual_style="modern",
            animation_cadence=AnimationCadence.MODERATE,
            mood="uplifting",
        )

        assert style_guide.palette == ["#FF0000", "#00FF00"]
        assert style_guide.visual_style == "modern"
        assert style_guide.animation_cadence == AnimationCadence.MODERATE
        assert style_guide.mood == "uplifting"

    def test_that_style_guide_serializes_to_dict(self) -> None:
        """Test StyleGuide serialization to dictionary."""
        style_guide = StyleGuide(
            palette=["#FF0000"],
            visual_style="minimalist",
            animation_cadence=AnimationCadence.SLOW,
            mood="calm",
        )

        serialized = style_guide.to_dict()

        assert serialized["palette"] == ["#FF0000"]
        assert serialized["visual_style"] == "minimalist"
        assert serialized["animation_cadence"] == "slow"
        assert serialized["mood"] == "calm"
