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

from lagom import Container
from unittest.mock import AsyncMock, patch

from parlant.adapters.nlp.emotion_analyzer import CerebrasEmotionAnalyzer
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.nlp.emotion import EmotionAnalysis


async def test_that_emotion_analyzer_returns_structured_analysis_with_all_required_fields(
    container: Container,
) -> None:
    logger = container[Logger]
    meter = container[Meter]

    analyzer = CerebrasEmotionAnalyzer(logger=logger, meter=meter)

    test_dialogue = "I'm so frustrated with this issue! It's absolutely terrible."

    with patch.object(
        analyzer, "_do_analyze", new_callable=AsyncMock
    ) as mock_analyze:
        mock_analyze.return_value = EmotionAnalysis(
            intensity=8,
            pacing="fast",
            focus_keywords=["frustrated", "terrible"],
            recommended_scene_duration=5000,
        )

        result = await analyzer.analyze(test_dialogue)

        assert isinstance(result, EmotionAnalysis)
        assert 1 <= result.intensity <= 10
        assert result.pacing in ("fast", "slow")
        assert isinstance(result.focus_keywords, list)
        assert all(isinstance(kw, str) for kw in result.focus_keywords)
        assert isinstance(result.recommended_scene_duration, int)
        assert result.recommended_scene_duration > 0


async def test_that_emotion_analyzer_handles_positive_emotions(
    container: Container,
) -> None:
    logger = container[Logger]
    meter = container[Meter]

    analyzer = CerebrasEmotionAnalyzer(logger=logger, meter=meter)

    positive_dialogue = "This is wonderful! I'm so happy and excited about this opportunity!"

    with patch.object(
        analyzer, "_do_analyze", new_callable=AsyncMock
    ) as mock_analyze:
        mock_analyze.return_value = EmotionAnalysis(
            intensity=7,
            pacing="fast",
            focus_keywords=["wonderful", "happy", "excited"],
            recommended_scene_duration=4000,
        )

        result = await analyzer.analyze(positive_dialogue)

        assert result.intensity >= 5
        assert result.pacing == "fast"
        assert any(kw in ["wonderful", "happy", "excited"] for kw in result.focus_keywords)


async def test_that_emotion_analyzer_handles_negative_emotions(
    container: Container,
) -> None:
    logger = container[Logger]
    meter = container[Meter]

    analyzer = CerebrasEmotionAnalyzer(logger=logger, meter=meter)

    negative_dialogue = "I'm extremely angry and disappointed. This is unacceptable!"

    with patch.object(
        analyzer, "_do_analyze", new_callable=AsyncMock
    ) as mock_analyze:
        mock_analyze.return_value = EmotionAnalysis(
            intensity=9,
            pacing="fast",
            focus_keywords=["angry", "disappointed", "unacceptable"],
            recommended_scene_duration=5500,
        )

        result = await analyzer.analyze(negative_dialogue)

        assert result.intensity >= 6
        assert result.pacing == "fast"
        assert any(
            kw in ["angry", "disappointed", "unacceptable"] for kw in result.focus_keywords
        )


async def test_that_emotion_analyzer_handles_neutral_emotions(
    container: Container,
) -> None:
    logger = container[Logger]
    meter = container[Meter]

    analyzer = CerebrasEmotionAnalyzer(logger=logger, meter=meter)

    neutral_dialogue = "I need to check my account balance. Can you help me with that?"

    with patch.object(
        analyzer, "_do_analyze", new_callable=AsyncMock
    ) as mock_analyze:
        mock_analyze.return_value = EmotionAnalysis(
            intensity=3,
            pacing="slow",
            focus_keywords=["check", "account"],
            recommended_scene_duration=3000,
        )

        result = await analyzer.analyze(neutral_dialogue)

        assert result.intensity <= 4
        assert result.pacing == "slow"
        assert isinstance(result.focus_keywords, list)


async def test_that_emotion_analyzer_returns_fallback_when_api_fails(
    container: Container,
) -> None:
    logger = container[Logger]
    meter = container[Meter]

    analyzer = CerebrasEmotionAnalyzer(logger=logger, meter=meter)

    test_dialogue = "I'm really frustrated with this!"

    with patch.object(analyzer, "_do_analyze", side_effect=Exception("API Error")):
        result = await analyzer.analyze(test_dialogue)

        assert isinstance(result, EmotionAnalysis)
        assert 1 <= result.intensity <= 10
        assert result.pacing in ("fast", "slow")
        assert isinstance(result.focus_keywords, list)
        assert isinstance(result.recommended_scene_duration, int)


async def test_that_emotion_analyzer_caches_repeated_analyses(
    container: Container,
) -> None:
    logger = container[Logger]
    meter = container[Meter]

    analyzer = CerebrasEmotionAnalyzer(logger=logger, meter=meter)

    test_dialogue = "I love this product!"

    with patch.object(
        analyzer, "_do_analyze", new_callable=AsyncMock
    ) as mock_analyze:
        mock_analyze.return_value = EmotionAnalysis(
            intensity=8,
            pacing="fast",
            focus_keywords=["love"],
            recommended_scene_duration=4000,
        )

        result1 = await analyzer.analyze(test_dialogue)
        result2 = await analyzer.analyze(test_dialogue)

        assert result1 == result2
        assert mock_analyze.call_count == 1
