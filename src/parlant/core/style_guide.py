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

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence


class AnimationCadence(str, Enum):
    """Enumeration of animation pacing options."""

    SLOW = "slow"
    MODERATE = "moderate"
    FAST = "fast"


class Color(str, Enum):
    """Common color options for visual design."""

    RED = "#FF0000"
    GREEN = "#00FF00"
    BLUE = "#0000FF"
    YELLOW = "#FFFF00"
    CYAN = "#00FFFF"
    MAGENTA = "#FF00FF"
    WHITE = "#FFFFFF"
    BLACK = "#000000"
    ORANGE = "#FFA500"
    PURPLE = "#800080"


@dataclass
class StyleGuide:
    """A style guide containing visual and aesthetic attributes for content generation."""

    palette: list[str]
    """List of hex color codes or color names defining the visual palette."""

    visual_style: str
    """The visual style descriptor (e.g., 'modern', 'minimalist', 'cyberpunk')."""

    animation_cadence: AnimationCadence
    """The pacing of animations and transitions."""

    mood: str
    """The emotional tone or atmosphere (e.g., 'uplifting', 'calm', 'energetic')."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the style guide to a dictionary.

        Returns:
            A dictionary representation suitable for JSON serialization.
        """
        return {
            "palette": self.palette,
            "visual_style": self.visual_style,
            "animation_cadence": self.animation_cadence.value,
            "mood": self.mood,
        }


def _extract_keywords_from_dialogues(dialogues: Sequence[str]) -> dict[str, int]:
    """Extract and count keywords from dialogue entries.

    Args:
        dialogues: Sequence of dialogue text entries.

    Returns:
        Dictionary mapping keywords to their frequency.
    """
    keyword_map: dict[str, int] = {}

    style_keywords = {
        "warm": ["warm", "sunset", "golden", "orange"],
        "cool": ["cool", "blue", "purple", "cold"],
        "calm": ["calm", "peaceful", "slow", "serene", "meditative"],
        "energetic": ["fast", "rapid", "dynamic", "vibrant", "action"],
        "minimalist": ["minimalist", "clean", "simple", "minimal"],
        "modern": ["modern", "contemporary", "professional", "sleek"],
        "cyberpunk": ["cyber", "neon", "futuristic", "digital"],
    }

    animation_keywords = {
        "slow": ["slow", "methodical", "steady", "gradual"],
        "moderate": ["moderate", "balanced", "even", "regular"],
        "fast": ["fast", "rapid", "quick", "swift", "dynamic"],
    }

    mood_keywords = {
        "uplifting": ["uplifting", "joyful", "happy", "bright"],
        "somber": ["somber", "dark", "gloomy", "melancholic"],
        "energetic": ["energetic", "exciting", "thrilling"],
        "contemplative": ["contemplative", "thoughtful", "reflective"],
        "peaceful": ["peaceful", "calm", "tranquil"],
    }

    combined_text = " ".join(dialogues).lower()

    for category, keywords in style_keywords.items():
        for keyword in keywords:
            if keyword in combined_text:
                keyword_map[f"style_{category}"] = keyword_map.get(f"style_{category}", 0) + 1

    for category, keywords in animation_keywords.items():
        for keyword in keywords:
            if keyword in combined_text:
                keyword_map[f"anim_{category}"] = keyword_map.get(f"anim_{category}", 0) + 1

    for category, keywords in mood_keywords.items():
        for keyword in keywords:
            if keyword in combined_text:
                keyword_map[f"mood_{category}"] = keyword_map.get(f"mood_{category}", 0) + 1

    return keyword_map


def _determine_palette(
    dialogues: Sequence[str], user_palette: Optional[list[str]] = None
) -> list[str]:
    """Determine the color palette based on dialogue analysis.

    Args:
        dialogues: Sequence of dialogue entries.
        user_palette: User-provided palette override.

    Returns:
        List of hex color codes.
    """
    if user_palette is not None:
        return user_palette

    color_keywords = {
        Color.ORANGE.value: ["orange", "warm", "sunset"],
        Color.PURPLE.value: ["purple", "violet", "lavender"],
        Color.BLUE.value: ["blue", "cool"],
        Color.BLACK.value: ["dark", "black", "shadowy"],
        Color.WHITE.value: ["white", "bright", "light"],
        Color.RED.value: ["red", "vibrant"],
    }

    combined_text = " ".join(dialogues).lower()
    palette: list[str] = []

    for color, keywords in color_keywords.items():
        for keyword in keywords:
            if keyword in combined_text and color not in palette:
                palette.append(color)

    if not palette:
        palette = [
            Color.BLUE.value,
            Color.WHITE.value,
            Color.BLACK.value,
        ]

    return palette


def _determine_visual_style(dialogues: Sequence[str], user_style: Optional[str] = None) -> str:
    """Determine visual style from dialogue analysis.

    Args:
        dialogues: Sequence of dialogue entries.
        user_style: User-provided style override.

    Returns:
        Visual style description.
    """
    if user_style is not None:
        return user_style

    combined_text = " ".join(dialogues).lower()

    style_patterns = {
        "minimalist": ["minimalist", "clean", "simple"],
        "modern": ["modern", "professional", "contemporary"],
        "cyberpunk": ["cyber", "futuristic", "neon"],
        "cinematic": ["cinematic", "film", "movie"],
        "artistic": ["artistic", "creative", "expressive"],
    }

    for style, keywords in style_patterns.items():
        if any(keyword in combined_text for keyword in keywords):
            return style

    return "balanced"


def _determine_animation_cadence(
    dialogues: Sequence[str], user_cadence: Optional[str] = None
) -> AnimationCadence:
    """Determine animation cadence from dialogue analysis.

    Args:
        dialogues: Sequence of dialogue entries.
        user_cadence: User-provided cadence override.

    Returns:
        AnimationCadence enumeration value.
    """
    if user_cadence is not None:
        try:
            return AnimationCadence(user_cadence.lower())
        except ValueError:
            return AnimationCadence.MODERATE

    combined_text = " ".join(dialogues).lower()

    if any(word in combined_text for word in ["slow", "methodical", "gradual", "steady"]):
        return AnimationCadence.SLOW

    if any(word in combined_text for word in ["fast", "rapid", "quick", "dynamic"]):
        return AnimationCadence.FAST

    return AnimationCadence.MODERATE


def _determine_mood(dialogues: Sequence[str], user_mood: Optional[str] = None) -> str:
    """Determine mood from dialogue analysis.

    Args:
        dialogues: Sequence of dialogue entries.
        user_mood: User-provided mood override.

    Returns:
        Mood description.
    """
    if user_mood is not None:
        return user_mood

    combined_text = " ".join(dialogues).lower()

    mood_patterns = {
        "uplifting": ["uplifting", "joyful", "happy", "bright", "positive"],
        "somber": ["somber", "dark", "gloomy", "melancholic", "sad"],
        "energetic": ["energetic", "exciting", "thrilling", "action"],
        "peaceful": ["peaceful", "calm", "tranquil", "serene"],
        "contemplative": ["contemplative", "thoughtful", "reflective"],
    }

    for mood, keywords in mood_patterns.items():
        if any(keyword in combined_text for keyword in keywords):
            return mood

    return "neutral"


async def generate_global_style_guide(
    dialogues: Sequence[str],
    user_overrides: Optional[Mapping[str, Any]] = None,
) -> StyleGuide:
    """Generate a global style guide from dialogues with optional user overrides.

    This function analyzes dialogue content to derive consistent visual and
    aesthetic attributes including palette, visual style, animation cadence,
    and mood. User-provided overrides take precedence over derived values.

    Args:
        dialogues: Sequence of dialogue entries to analyze.
        user_overrides: Optional mapping of style attributes to override.
                       Supported keys: palette, visual_style, animation_cadence, mood.

    Returns:
        A StyleGuide instance with all attributes populated.
    """
    user_overrides = user_overrides or {}

    palette = _determine_palette(dialogues, user_palette=user_overrides.get("palette"))
    visual_style = _determine_visual_style(dialogues, user_style=user_overrides.get("visual_style"))
    animation_cadence = _determine_animation_cadence(
        dialogues, user_cadence=user_overrides.get("animation_cadence")
    )
    mood = _determine_mood(dialogues, user_mood=user_overrides.get("mood"))

    return StyleGuide(
        palette=palette,
        visual_style=visual_style,
        animation_cadence=animation_cadence,
        mood=mood,
    )
