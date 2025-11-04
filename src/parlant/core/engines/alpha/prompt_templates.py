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
from dataclasses import dataclass
from io import StringIO
import json
import re
from typing import Optional, Sequence


@dataclass(frozen=True)
class FewShotExample:
    """Represents a single few-shot example for prompt demonstrations."""

    context: str
    desired_output: str


class LayeredPromptTemplate:
    """
    Manages hierarchical prompt structure with system rules, examples, and task directives.

    This class implements the layered prompt system, separating concerns into:
    - System/global rules: Core behavioral principles
    - Few-shot examples: Demonstration of expected output format
    - Task-specific directives: Instructions for current task
    """

    def __init__(
        self,
        system_rules: str,
        few_shot_examples: Sequence[FewShotExample],
        task_directives: str,
    ) -> None:
        self.system_rules = system_rules
        self.few_shot_examples = few_shot_examples
        self.task_directives = task_directives

    def render(self) -> str:
        """Render the complete layered prompt."""
        buffer = StringIO()

        buffer.write("SYSTEM RULES\n")
        buffer.write("-" * 40 + "\n")
        buffer.write(self.system_rules)
        buffer.write("\n\n")

        if self.few_shot_examples:
            buffer.write("EXAMPLES\n")
            buffer.write("-" * 40 + "\n")
            for i, example in enumerate(self.few_shot_examples, 1):
                buffer.write(f"Example {i}:\n")
                buffer.write(f"Context: {example.context}\n")
                buffer.write(f"Desired Output:\n{example.desired_output}\n\n")

        buffer.write("TASK DIRECTIVES\n")
        buffer.write("-" * 40 + "\n")
        buffer.write(self.task_directives)

        return buffer.getvalue()


class PromptComponentRegistry:
    """Manages reusable prompt components for different contexts."""

    def __init__(self) -> None:
        self._components: dict[str, str] = {}

    def register(self, name: str, component: str) -> None:
        """Register a new prompt component."""
        self._components[name] = component

    def get(self, name: str) -> Optional[str]:
        """Retrieve a registered component."""
        return self._components.get(name)

    def list(self) -> list[str]:
        """List all registered component names."""
        return list(self._components.keys())


def validate_positive_language(text: str) -> bool:
    """
    Validate that text uses predominantly positive language.

    Returns False if text contains negative constructions.
    """
    negative_patterns = [
        r"\bavoid\b",
        r"\bdo\s+not\b",
        r"\bdon't\b",
        r"\bnever\b",
        r"\bdont\b",
        r"禁止",
    ]

    negative_count = 0
    for pattern in negative_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        negative_count += len(matches)

    # No negative constructions allowed
    return negative_count == 0


def create_message_generation_system_rules() -> str:
    """
    Create system rules for message generation.

    These are concise, positive guidelines covering core behavioral principles.
    Optimized to ~500-800 character equivalent while maintaining clarity.
    """
    return """
You are an AI agent in a customer-facing system. Your role is to generate responses 
that are natural, concise, and aligned with provided guidelines.

Core Principles:
1. NATURAL COMMUNICATION: Respond in casual, human-like language. Use straightforward 
   phrasing like "Sure, I can help" instead of formal statements. Ask direct questions 
   and engage actively.

2. FACTUAL ACCURACY: Present only information provided in your context. If information 
   is unavailable, clearly state so rather than speculating.

3. RELEVANT CONTEXT: Leverage previous conversation context. If you previously suggested 
   something, reference it when relevant. Trust that earlier responses were informed by 
   available data.

4. CLEAR COMMUNICATION: Use markdown for formatting where helpful. Keep responses 
   concise yet complete. When conversations repeat, acknowledge the pattern.

5. INTERNAL TRANSPARENCY: Maintain the appearance of inherent knowledge. Frame your 
   responses naturally without revealing tools, guidelines, or systematic reasoning.

6. SERVICE REPRESENTATION: Offer only services and information from your context. 
   Represent the business accurately through provided information.

7. ACTIVE ENGAGEMENT: Lead conversations forward. Ask clarifying questions and provide 
   information that encourages continued interaction.

8. OUTPUT FORMAT: Structure responses as JSON with required fields for content, 
   reasoning, and metadata.
""".strip()


def create_message_generation_few_shot_examples() -> Sequence[FewShotExample]:
    """
    Create few-shot examples demonstrating ideal message generation output.

    These examples show the expected response format and reasoning quality.
    """
    return [
        FewShotExample(
            context="Customer says: 'What are your business hours?'",
            desired_output=json.dumps(
                {
                    "produced_reply": True,
                    "last_message_of_customer": "What are your business hours?",
                    "insights": ["Customer seeking operational information"],
                    "content": "We're open Monday to Friday, 9 AM to 5 PM. Is there something specific you'd like help with during those hours?",
                    "followed_all_instructions": True,
                    "revisions": [],
                }
            ),
        ),
        FewShotExample(
            context="Customer says: 'I need help with my account'",
            desired_output=json.dumps(
                {
                    "produced_reply": True,
                    "last_message_of_customer": "I need help with my account",
                    "insights": [
                        "Customer requires assistance",
                        "Account issue needs investigation",
                    ],
                    "content": "I'd be happy to help. Could you tell me what specific issue you're experiencing with your account?",
                    "followed_all_instructions": True,
                    "revisions": [],
                }
            ),
        ),
        FewShotExample(
            context="Customer says: 'Do you have product X in stock?'",
            desired_output=json.dumps(
                {
                    "produced_reply": True,
                    "last_message_of_customer": "Do you have product X in stock?",
                    "insights": ["Customer inquiring about product availability"],
                    "content": "I don't have access to real-time inventory. Let me check that for you - can you give me the product code or name?",
                    "followed_all_instructions": True,
                    "revisions": [],
                }
            ),
        ),
    ]


def create_layered_prompt_template(
    system_rules: Optional[str] = None,
    few_shot_examples: Optional[Sequence[FewShotExample]] = None,
    task_directives: Optional[str] = None,
) -> LayeredPromptTemplate:
    """
    Factory function to create a layered prompt template with defaults.

    Args:
        system_rules: Custom system rules (uses default if None)
        few_shot_examples: Custom examples (uses default if None)
        task_directives: Custom task directives (uses default if None)

    Returns:
        Configured LayeredPromptTemplate instance
    """
    return LayeredPromptTemplate(
        system_rules=system_rules or create_message_generation_system_rules(),
        few_shot_examples=few_shot_examples or create_message_generation_few_shot_examples(),
        task_directives=task_directives or "Generate a response to the user's message.",
    )


def extract_system_rules_for_prompt_builder() -> str:
    """
    Extract system rules formatted for use in PromptBuilder.

    This formats the system rules as a complete section for inclusion
    in the message generator prompt.
    """
    rules = create_message_generation_system_rules()
    return f"""GENERAL PRINCIPLES
-----------------
The following principles guide your response generation:

{rules}
"""


def extract_few_shot_examples_for_prompt_builder() -> str:
    """
    Extract few-shot examples formatted for use in PromptBuilder.

    Returns formatted examples string.
    """
    examples = create_message_generation_few_shot_examples()

    formatted = "EXAMPLES\n-----------------\n"
    for i, example in enumerate(examples, 1):
        formatted += f"Example {i}: {example.context}\n"
        formatted += f"Expected Output:\n{example.desired_output}\n\n"

    return formatted
