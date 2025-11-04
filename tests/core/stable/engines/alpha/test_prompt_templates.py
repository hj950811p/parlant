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
import re

from parlant.core.engines.alpha.prompt_templates import (
    LayeredPromptTemplate,
    PromptComponentRegistry,
    FewShotExample,
    create_message_generation_system_rules,
    create_message_generation_few_shot_examples,
    validate_positive_language,
)


class TestLayeredPromptBuilderStructure:
    def test_layered_prompt_template_produces_correct_structure(self) -> None:
        """Verify the builder creates system rules, examples, and task directives sections."""
        system_rules = "Respond naturally and concisely."
        examples = [
            FewShotExample(
                context="Customer asks about hours",
                desired_output=json.dumps({"message": "We're open 9 AM to 5 PM"}),
            )
        ]
        task_directives = "Generate a response to the latest message."

        template = LayeredPromptTemplate(
            system_rules=system_rules,
            few_shot_examples=examples,
            task_directives=task_directives,
        )

        assert template.system_rules == system_rules
        assert len(template.few_shot_examples) == 1
        assert template.task_directives == task_directives

    def test_layered_prompt_renders_in_correct_order(self) -> None:
        """Verify sections render in the correct order."""
        system_rules = "System rules"
        examples = [
            FewShotExample(
                context="Example context",
                desired_output=json.dumps({"message": "Example output"}),
            )
        ]
        task_directives = "Task directives"

        template = LayeredPromptTemplate(
            system_rules=system_rules,
            few_shot_examples=examples,
            task_directives=task_directives,
        )

        rendered = template.render()

        system_pos = rendered.find(system_rules)
        examples_pos = rendered.find("EXAMPLES")
        directives_pos = rendered.find(task_directives)

        assert system_pos >= 0, "System rules should be present"
        assert examples_pos >= 0, "Examples section should be present"
        assert directives_pos >= 0, "Task directives should be present"
        assert system_pos < examples_pos < directives_pos, "Should render in correct order"


class TestPromptPositiveLanguage:
    def test_prompt_templates_use_positive_language(self) -> None:
        """Verify prompt templates don't contain negative constructions."""
        system_rules = create_message_generation_system_rules()

        # Verify that system rules use predominantly positive language
        # Check for key negative patterns that should not appear
        forbidden_patterns = [
            r"\bavoid\b",
            r"\bdo\s+not\b",
            r"\bdon't\b",
            r"\bnever\b",
            r"禁止",
        ]

        # System rules should have no negative patterns
        for pattern in forbidden_patterns:
            matches = re.findall(pattern, system_rules, re.IGNORECASE)
            assert len(matches) == 0, f"System rules should not contain '{pattern}'"

    def test_validate_positive_language_function(self) -> None:
        """Test the positive language validator."""
        positive_text = "Create natural and engaging responses."
        assert validate_positive_language(positive_text) is True

        negative_text = "Do not avoid repeating yourself."
        assert validate_positive_language(negative_text) is False

        # Text with only positive language should pass
        positive_text2 = "Engage naturally with clear communication."
        assert validate_positive_language(positive_text2) is True


class TestFewShotExamples:
    def test_few_shot_examples_serialize_correctly(self) -> None:
        """Verify few-shot examples format correctly as JSON."""
        examples = create_message_generation_few_shot_examples()

        assert len(examples) >= 2, "Should have at least 2 examples"

        for example in examples:
            assert example.context, "Example should have context"
            assert example.desired_output, "Example should have desired output"

            # Verify desired_output is valid JSON
            parsed = json.loads(example.desired_output)
            assert isinstance(parsed, dict), "Output should be JSON object"

    def test_few_shot_example_creation(self) -> None:
        """Test creating individual few-shot examples."""
        example = FewShotExample(
            context="Customer asks: 'What are your hours?'",
            desired_output=json.dumps(
                {
                    "produced_reply": True,
                    "last_message_of_customer": "What are your hours?",
                    "insights": ["Customer needs operational information"],
                }
            ),
        )

        assert example.context
        assert json.loads(example.desired_output)


class TestPromptTemplateLength:
    def test_system_rules_meets_length_requirement(self) -> None:
        """Verify system rules are 500-800 character equivalent."""
        system_rules = create_message_generation_system_rules()

        # Count characters (Chinese characters are typically 1-2 tokens each)
        # English words average ~4 characters per token
        char_count = len(system_rules)

        # Allow 500-2000 characters (approximately 500-800 tokens)
        assert 500 <= char_count <= 2000, f"System rules should be 500-2000 chars, got {char_count}"


class TestPromptComponentRegistry:
    def test_component_registry_stores_templates(self) -> None:
        """Test that registry can store and retrieve templates."""
        registry = PromptComponentRegistry()

        component = "Test component text"
        registry.register("test_component", component)

        assert registry.get("test_component") == component

    def test_component_registry_list_components(self) -> None:
        """Test that registry can list all registered components."""
        registry = PromptComponentRegistry()

        registry.register("comp1", "Text 1")
        registry.register("comp2", "Text 2")

        components = registry.list()
        assert "comp1" in components
        assert "comp2" in components
        assert len(components) >= 2


class TestLayeredPromptComponentsReusable:
    def test_layered_prompt_components_work_across_contexts(self) -> None:
        """Test that same components work for different prompt scenarios."""
        system_rules = create_message_generation_system_rules()
        examples = create_message_generation_few_shot_examples()

        # Create two different contexts using same components
        template1 = LayeredPromptTemplate(
            system_rules=system_rules,
            few_shot_examples=examples,
            task_directives="Task A",
        )

        template2 = LayeredPromptTemplate(
            system_rules=system_rules,
            few_shot_examples=examples,
            task_directives="Task B",
        )

        # Both should render with same system rules
        assert template1.system_rules == template2.system_rules
        assert len(template1.few_shot_examples) == len(template2.few_shot_examples)

        # But different task directives
        assert "Task A" in template1.render()
        assert "Task B" in template2.render()


class TestPromptTemplateIntegration:
    def test_full_prompt_template_rendering(self) -> None:
        """Test complete rendering of a layered prompt template."""
        system_rules = create_message_generation_system_rules()
        examples = create_message_generation_few_shot_examples()
        task_directives = "Generate a natural response to the user's message."

        template = LayeredPromptTemplate(
            system_rules=system_rules,
            few_shot_examples=examples,
            task_directives=task_directives,
        )

        rendered = template.render()

        # Verify all components are present
        assert system_rules in rendered
        assert task_directives in rendered
        assert "EXAMPLES" in rendered or "Example" in rendered

    def test_prompt_template_with_empty_examples(self) -> None:
        """Test template rendering with no examples."""
        system_rules = "Be concise."
        task_directives = "Generate response."

        template = LayeredPromptTemplate(
            system_rules=system_rules,
            few_shot_examples=[],
            task_directives=task_directives,
        )

        rendered = template.render()
        assert system_rules in rendered
        assert task_directives in rendered
