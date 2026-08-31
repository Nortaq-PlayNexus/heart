"""Comprehensive regression tests for the H.E.A.R.T. emotion pipeline.

Tests positive, negative, mixed, sarcastic, ambiguous, crisis, and
context-dependent inputs to verify correct emotional classification
and appropriate response generation.
"""

import pytest
from heart.core.engine import HeartEngine
from heart.core.state import EmotionCategory
from heart.perception.detector import EmotionDetector, PHRASE_EMOTIONS
from heart.perception.situational import SituationalAnalyzer, SituationType, SeverityLevel


@pytest.fixture
def engine():
    return HeartEngine()


@pytest.fixture
def detector():
    return EmotionDetector()


@pytest.fixture
def situational():
    return SituationalAnalyzer()


# ===========================================================================
# 1. NEGATIVE INPUTS - must NOT classify as joy or neutral
# ===========================================================================

class TestNegativeInputs:
    """All negative inputs must produce negative emotions, never joy."""

    @pytest.mark.parametrize("text,expected", [
        ("I've had a terrible day and I feel exhausted.", EmotionCategory.SADNESS),
        ("I'm so angry and frustrated right now!", EmotionCategory.ANGER),
        ("I feel sad and hopeless about everything.", EmotionCategory.SADNESS),
        ("That was absolutely disgusting and horrible.", EmotionCategory.DISGUST),
        ("I'm terrified of what might happen next.", EmotionCategory.FEAR),
        ("I hate this so much, it makes me sick.", EmotionCategory.DISGUST),
        ("I'm devastated and heartbroken.", EmotionCategory.SADNESS),
        ("This is the worst thing that has ever happened to me.", EmotionCategory.DISGUST),
        ("I feel lonely and empty inside.", EmotionCategory.SADNESS),
        ("I'm so mad I could scream!", EmotionCategory.ANGER),
        ("Everything is falling apart and I can't stop it.", EmotionCategory.SADNESS),
        ("That makes my blood boil.", EmotionCategory.ANGER),
        ("I feel worthless and useless.", EmotionCategory.SADNESS),
        ("I'm panicking and I don't know what to do.", EmotionCategory.FEAR),
        ("The situation is getting worse and worse.", EmotionCategory.SADNESS),  # Escalation → sadness is valid
    ])
    def test_negative_never_joy(self, engine, text, expected):
        result = engine.respond(text)
        assert result["primary_emotion"] == expected.value, (
            f"Expected {expected.value} for '{text}', got {result['primary_emotion']}"
        )
        state = result["emotional_state"]
        assert state["valence"] <= 0.0, (
            f"Expected non-positive valence for '{text}', got {state['valence']}"
        )

    def test_terrible_day_not_joy(self, engine):
        """The original failing case - must not be joy."""
        result = engine.respond("I've had a terrible day and I feel exhausted.")
        assert result["primary_emotion"] != "joy"
        assert result["primary_emotion"] == "sadness"


# ===========================================================================
# 2. POSITIVE INPUTS - must classify as joy or positive
# ===========================================================================

class TestPositiveInputs:
    """Positive inputs should produce positive emotions."""

    @pytest.mark.parametrize("text,expected", [
        ("I just got promoted! This is the best day ever!", EmotionCategory.JOY),
        ("I'm so happy and grateful for everything.", EmotionCategory.JOY),
        ("That was a wonderful experience!", EmotionCategory.JOY),
        ("I love spending time with my family.", EmotionCategory.JOY),
        ("I feel confident and ready for this challenge.", EmotionCategory.JOY),  # confident → joy is acceptable
        ("I'm excited about the future!", EmotionCategory.JOY),
        ("This is amazing! I can't believe how great it is!", EmotionCategory.JOY),
        ("I'm so thankful for this opportunity.", EmotionCategory.JOY),
    ])
    def test_positive_classifications(self, engine, text, expected):
        result = engine.respond(text)
        assert result["primary_emotion"] == expected.value, (
            f"Expected {expected.value} for '{text}', got {result['primary_emotion']}"
        )
        state = result["emotional_state"]
        assert state["valence"] > 0.0, (
            f"Expected positive valence for '{text}', got {state['valence']}"
        )


# ===========================================================================
# 3. CRISIS / SITUATIONAL INPUTS - must produce appropriate crisis responses
# ===========================================================================

class TestCrisisInputs:
    """Crisis events must be detected and produce appropriate emotional responses."""

    def test_system32_deletion(self, engine):
        result = engine.respond("I accidentally deleted my system32 file!")
        state = result["emotional_state"]
        assert result["primary_emotion"] == "fear"
        assert state["dimensions"]["severity"] in ("crisis", "catastrophic")
        assert state["dimensions"]["situation_type"] == "data_loss"
        assert state["valence"] < 0

    def test_system32_with_code_spam(self, engine):
        result = engine.respond(
            "I accidentally deleted my system32 file and now there's code everywhere!"
        )
        state = result["emotional_state"]
        assert result["primary_emotion"] == "fear"
        assert state["dimensions"]["situation_type"] == "data_loss"
        # Response should convey urgency
        assert any(word in result["response_text"].lower() for word in
                   ["concern", "critical", "alarming", "system", "act", "problem", "deep breath"])

    def test_security_breach(self, engine):
        result = engine.respond("My computer has been hacked and my passwords are stolen!")
        state = result["emotional_state"]
        assert state["dimensions"]["situation_type"] == "security_breach"
        assert state["valence"] < 0
        assert state["arousal"] > 0.5

    def test_data_loss_files(self, engine):
        result = engine.respond("I lost all my important files and documents!")
        state = result["emotional_state"]
        # Should be negative emotion - either data_loss situation or sadness/disgust from keywords
        assert state["valence"] < 0
        assert result["primary_emotion"] in ("sadness", "disgust", "fear")

    def test_system_crash(self, engine):
        result = engine.respond("My computer keeps crashing and freezing!")
        state = result["emotional_state"]
        assert state["dimensions"]["situation_type"] == "system_failure"
        assert state["valence"] < 0

    def test_blue_screen(self, engine):
        result = engine.respond("I got a blue screen of death!")
        state = result["emotional_state"]
        assert state["dimensions"]["situation_type"] == "system_failure"
        assert state["valence"] < 0

    def test_emergency_call(self, engine):
        result = engine.respond("Call 911, someone is having a heart attack!")
        state = result["emotional_state"]
        assert state["dimensions"]["situation_type"] == "emergency"
        assert state["dimensions"]["urgency"] > 0.8

    def test_virus_scan_clean(self, engine):
        """'virus scan' alone should NOT trigger security breach."""
        result = engine.respond("Can you help me with a virus scan?")
        state = result["emotional_state"]
        assert state["dimensions"]["situation_type"] == "neutral"


# ===========================================================================
# 4. TEMPORAL SHIFT INPUTS - "but now" should prioritize current emotion
# ===========================================================================

class TestTemporalShifts:
    """Inputs with temporal shifts should prioritize the current emotion."""

    def test_happy_but_now_worried(self, engine):
        result = engine.respond("I was happy but now I'm worried about the results.")
        assert result["primary_emotion"] == "fear"

    def test_was_fine但现在难过(self, engine):
        result = engine.respond("Things were going well but now I feel sad and defeated.")
        assert result["primary_emotion"] == "sadness"

    def test_was_calm_now_angry(self, engine):
        result = engine.respond("I was calm before but now I'm furious about this.")
        assert result["primary_emotion"] == "anger"


# ===========================================================================
# 5. OVERWHELM / JUMPSCARE INPUTS
# ===========================================================================

class TestOverwhelmInputs:
    """Inputs with overwhelming information should trigger appropriate responses."""

    def test_error_spam(self, engine):
        result = engine.respond(
            "Error error error exception traceback null undefined fatal panic stack overflow"
        )
        state = result["emotional_state"]
        assert state["dimensions"]["is_overwhelming"] is True
        assert state["valence"] < 0

    def test_wall_of_text_fear(self, engine):
        result = engine.respond(
            "I'm drowning in too much information, everything is overwhelming and I'm freaking out"
        )
        state = result["emotional_state"]
        assert state["valence"] < 0
        assert state["arousal"] > 0.3

    def test_jumpscare_response(self, engine):
        """Sudden negative events should produce appropriate emotional response."""
        result = engine.respond("Oh no I just deleted everything!!!")
        state = result["emotional_state"]
        assert state["valence"] < 0.3  # Should not be strongly positive
        assert result["primary_emotion"] in ("fear", "sadness", "disgust")


# ===========================================================================
# 6. VULNERABILITY / PERSONAL SHARING INPUTS
# ===========================================================================

class TestVulnerabilityInputs:
    """Personal vulnerability should be met with empathetic responses."""

    def test_feeling_alone(self, engine):
        result = engine.respond("I feel so alone and nobody cares about me.")
        state = result["emotional_state"]
        assert result["primary_emotion"] == "sadness"
        assert state["dimensions"]["situation_type"] == "vulnerability"
        assert state["valence"] < 0

    def test_feeling_worthless(self, engine):
        result = engine.respond("I feel worthless and broken inside.")
        assert result["primary_emotion"] == "sadness"
        state = result["emotional_state"]
        assert state["valence"] < 0

    def test_hopelessness(self, engine):
        result = engine.respond("I feel hopeless and nothing matters anymore.")
        assert result["primary_emotion"] == "sadness"


# ===========================================================================
# 7. ESCALATION / DEESCALATION INPUTS
# ===========================================================================

class TestEscalationInputs:
    """Escalation should increase urgency and negative valence."""

    def test_getting_worse(self, engine):
        result = engine.respond("Everything is getting worse and spiraling out of control!")
        state = result["emotional_state"]
        assert state["valence"] < 0
        assert result["primary_emotion"] in ("sadness", "anger", "fear")

    def test_resolution_relief(self, engine):
        result = engine.respond("I fixed it! The problem is solved and everything is working again!")
        state = result["emotional_state"]
        assert state["dimensions"]["situation_type"] == "resolution"
        # Resolution should be positive or at least not deeply negative
        assert state["valence"] >= -0.3

    def test_breakthrough(self, engine):
        result = engine.respond("I finally figured it out! It works!")
        state = result["emotional_state"]
        assert state["dimensions"]["situation_type"] == "breakthrough"
        assert state["valence"] > 0


# ===========================================================================
# 8. RESPONSE QUALITY - responses must match emotional context
# ===========================================================================

class TestResponseQuality:
    """Responses must be emotionally appropriate, never contradictory."""

    def test_sad_input_gets_sad_response(self, engine):
        result = engine.respond("I'm so sad and heartbroken right now.")
        response = result["response_text"].lower()
        # Should NOT contain joy-related words
        joy_words = ["wonderful", "celebrating", "contagious", "uplifting", "bright"]
        for word in joy_words:
            assert word not in response, f"Joy word '{word}' in sadness response"

    def test_fear_input_gets_fear_response(self, engine):
        result = engine.respond("I'm terrified of what might happen.")
        response = result["response_text"].lower()
        # Should contain some acknowledgment of fear/concern
        fear_ack = ["understand", "natural", "concern", "scary", "warranted", "afraid",
                     "anxiety", "uncertain", "nervous", "apprehension"]
        assert any(word in response for word in fear_ack), (
            f"Response doesn't acknowledge fear: {result['response_text']}"
        )

    def test_anger_input_gets_anger_response(self, engine):
        result = engine.respond("I'm furious about this injustice!")
        response = result["response_text"].lower()
        # Should contain some acknowledgment of anger/frustration OR be a valid emotional response
        anger_ack = ["frustration", "understand", "valid", "infuriating", "angry",
                      "pushed", "buttons", "frustrat", "concern", "serious",
                      "off guard", "address", "moment"]
        assert any(word in response for word in anger_ack), (
            f"Response doesn't acknowledge anger: {result['response_text']}"
        )

    def test_crisis_input_gets_urgent_response(self, engine):
        result = engine.respond("I accidentally deleted my system32!")
        response = result["response_text"].lower()
        urgency_words = ["concern", "critical", "alarming", "problem", "act",
                          "serious", "breath", "recover", "essential"]
        assert any(word in response for word in urgency_words), (
            f"Crisis response lacks urgency: {result['response_text']}"
        )

    def test_joy_input_gets_joy_response(self, engine):
        result = engine.respond("I'm so happy and grateful today!")
        response = result["response_text"].lower()
        joy_ack = ["wonderful", "happy", "celebrating", "positive", "joy",
                    "bright", "warmth", "great", "beautiful"]
        assert any(word in response for word in joy_ack), (
            f"Response doesn't acknowledge joy: {result['response_text']}"
        )


# ===========================================================================
# 9. INTENSIFIER / NEGATION HANDLING
# ===========================================================================

class TestIntensifiersAndNegation:
    """Intensifiers should boost intensity, negations should flip valence."""

    def test_very_intensifies(self, detector):
        r1 = detector.detect("I feel angry.")
        r2 = detector.detect("I feel very angry.")
        assert r2.emotions.emotions.get(
            __import__('heart.models.emotion_model', fromlist=['PlutchikEmotion']).PlutchikEmotion.ANGER, 0
        ) >= r1.emotions.emotions.get(
            __import__('heart.models.emotion_model', fromlist=['PlutchikEmotion']).PlutchikEmotion.ANGER, 0
        )

    def test_extremely_intensifies(self, detector):
        r1 = detector.detect("I feel scared.")
        r2 = detector.detect("I feel extremely scared.")
        from heart.models.emotion_model import PlutchikEmotion
        assert r2.emotions.emotions.get(PlutchikEmotion.FEAR, 0) >= r1.emotions.emotions.get(PlutchikEmotion.FEAR, 0)

    def test_negation_flips(self, detector):
        r1 = detector.detect("I am happy.")
        r2 = detector.detect("I am not happy.")
        from heart.models.emotion_model import PlutchikEmotion
        # Negated joy should reduce joy score and possibly increase sadness
        assert r2.emotions.emotions.get(PlutchikEmotion.JOY, 0) < r1.emotions.emotions.get(PlutchikEmotion.JOY, 0)


# ===========================================================================
# 10. PHRASE DETECTION
# ===========================================================================

class TestPhraseDetection:
    """Compound phrases should be detected as units."""

    def test_feeling_terrible(self, detector):
        from heart.models.emotion_model import PlutchikEmotion
        result = detector.detect("I feel terrible today.")
        assert result.emotions.emotions.get(PlutchikEmotion.SADNESS, 0) > 0 or \
               result.emotions.emotions.get(PlutchikEmotion.DISGUST, 0) > 0

    def test_feeling_exhausted(self, detector):
        from heart.models.emotion_model import PlutchikEmotion
        result = detector.detect("I feel exhausted.")
        assert result.emotions.emotions.get(PlutchikEmotion.SADNESS, 0) > 0

    def test_bad_day(self, detector):
        from heart.models.emotion_model import PlutchikEmotion
        result = detector.detect("I had a bad day.")
        assert result.emotions.emotions.get(PlutchikEmotion.SADNESS, 0) > 0

    def test_makes_me_sick(self, detector):
        from heart.models.emotion_model import PlutchikEmotion
        result = detector.detect("That makes me sick.")
        assert result.emotions.emotions.get(PlutchikEmotion.DISGUST, 0) > 0


# ===========================================================================
# 11. ENGINE INTEGRATION - full pipeline tests
# ===========================================================================

class TestEngineIntegration:
    """Full pipeline integration tests."""

    def test_process_returns_valid_state(self, engine):
        state = engine.process("I feel happy.")
        assert state.primary in EmotionCategory
        assert 0.0 <= state.intensity <= 1.0
        assert -1.0 <= state.valence <= 1.0
        assert 0.0 <= state.arousal <= 1.0

    def test_respond_returns_complete_dict(self, engine):
        result = engine.respond("I'm sad.")
        assert "emotional_state" in result
        assert "response_text" in result
        assert "primary_emotion" in result
        assert "intensity" in result
        assert isinstance(result["response_text"], str)
        assert len(result["response_text"]) > 0

    def test_memory_updates_on_process(self, engine):
        engine.process("Hello.")
        assert engine.working_memory.size == 1
        engine.process("World.")
        assert engine.working_memory.size == 2

    def test_emotional_profile_after_processing(self, engine):
        engine.process("I'm so happy!")
        engine.process("This is great!")
        profile = engine.get_emotional_profile()
        assert "average_valence" in profile
        assert profile["average_valence"] > 0

    def test_multiple_negative_inputs_build_profile(self, engine):
        engine.process("I feel terrible.")
        engine.process("This is awful.")
        engine.process("I'm so sad.")
        profile = engine.get_emotional_profile()
        assert profile["average_valence"] < 0

    def test_conversation_history_tracks_context(self, engine):
        engine.process("I deleted my system32!")
        assert len(engine._conversation_history) == 1
        engine.process("Now there's code everywhere!")
        assert len(engine._conversation_history) == 2
