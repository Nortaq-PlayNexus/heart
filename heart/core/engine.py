"""Core H.E.A.R.T. engine with situational awareness and reactive emotions.

The engine doesn't just detect emotions in text -- it FEELS the situation.
It understands events, crises, urgency, and context, and produces emotional
responses that reflect genuine reactions to what's happening.
"""

import re
from datetime import datetime, timezone
from typing import Optional
from heart.core.config import HeartConfig
from heart.core.state import EmotionalState, EmotionCategory, CompositeEmotion
from heart.models.emotion_model import (
    EmotionModel,
    DimensionalModel,
    PlutchikModel,
    PlutchikEmotion,
    EmotionScores,
)
from heart.perception.detector import EmotionDetector
from heart.perception.analyzer import SentimentAnalyzer
from heart.perception.features import FeatureExtractor
from heart.perception.situational import (
    SituationalAnalyzer,
    SituationType,
    SeverityLevel,
)
from heart.memory.working import WorkingMemory
from heart.memory.episodic import EpisodicMemory
from heart.memory.semantic import SemanticMemory
from heart.models.personality import PersonalityProfile, BigFiveTrait
from heart.response.generator import ResponseGenerator


# Maps situation types to emotional responses the system "feels"
_SITUATION_EMOTION_MAP: dict[SituationType, tuple[PlutchikEmotion, float, float]] = {
    # (emotion, valence, arousal) -- what the system feels in response
    SituationType.NEUTRAL: (PlutchikEmotion.TRUST, 0.1, 0.2),
    SituationType.TASK_IN_PROGRESS: (PlutchikEmotion.ANTICIPATION, 0.2, 0.4),
    SituationType.PROBLEM_REPORTED: (PlutchikEmotion.FEAR, -0.3, 0.5),
    SituationType.CRISIS_EVENT: (PlutchikEmotion.FEAR, -0.8, 0.9),
    SituationType.DATA_LOSS: (PlutchikEmotion.FEAR, -0.9, 0.9),  # Panic, not just sadness
    SituationType.SECURITY_BREACH: (PlutchikEmotion.FEAR, -0.85, 0.95),
    SituationType.SYSTEM_FAILURE: (PlutchikEmotion.FEAR, -0.7, 0.8),
    SituationType.EMERGENCY: (PlutchikEmotion.FEAR, -0.95, 1.0),
    SituationType.RESOLUTION: (PlutchikEmotion.JOY, 0.7, 0.5),
    SituationType.ESCALATION: (PlutchikEmotion.ANGER, -0.6, 0.8),
    SituationType.DEESCALATION: (PlutchikEmotion.TRUST, 0.3, 0.3),
    SituationType.MISUNDERSTANDING: (PlutchikEmotion.SURPRISE, -0.1, 0.5),
    SituationType.PERSONAL_SHARING: (PlutchikEmotion.TRUST, 0.2, 0.3),
    SituationType.JOINT_CELEBRATION: (PlutchikEmotion.JOY, 0.9, 0.8),
    SituationType.CONFLICT: (PlutchikEmotion.ANGER, -0.5, 0.7),
    SituationType.VULNERABILITY: (PlutchikEmotion.SADNESS, -0.7, 0.4),
    SituationType.OVERWHELM: (PlutchikEmotion.FEAR, -0.5, 0.8),
    SituationType.BREAKTHROUGH: (PlutchikEmotion.JOY, 0.8, 0.7),
    SituationType.SETBACK: (PlutchikEmotion.SADNESS, -0.6, 0.6),
}

# Severity multipliers
_SEVERITY_MULTIPLIER: dict[SeverityLevel, float] = {
    SeverityLevel.NORMAL: 1.0,
    SeverityLevel.LOW: 1.1,
    SeverityLevel.MODERATE: 1.3,
    SeverityLevel.HIGH: 1.5,
    SeverityLevel.CRISIS: 1.8,
    SeverityLevel.CATASTROPHIC: 2.0,
}


class HeartEngine:
    """The core H.E.A.R.T. engine.

    Processes text through multiple layers:
    1. Feature extraction (text characteristics)
    2. Emotion detection (keyword/phrase matching)
    3. Situational analysis (event/crisis understanding)
    4. Emotional state calculation (combining signals)
    5. Memory integration (storing experience)
    6. Response generation (emotionally appropriate output)

    The engine produces emotional STATES, not just classifications.
    It reacts to situations with genuine-feeling emotions.
    """

    def __init__(self, config: Optional[HeartConfig] = None):
        self.config = config or HeartConfig()
        self.emotion_model = EmotionModel()
        self.dimensional_model = DimensionalModel()
        self.plutchik_model = PlutchikModel()
        self.detector = EmotionDetector()
        self.analyzer = SentimentAnalyzer()
        self.extractor = FeatureExtractor()
        self.situational = SituationalAnalyzer()
        self.working_memory = WorkingMemory(capacity=self.config.working_memory_size)
        self.episodic_memory = EpisodicMemory(capacity=self.config.memory_capacity)
        self.semantic_memory = SemanticMemory()
        self.personality = PersonalityProfile()
        self.response_generator = ResponseGenerator()
        self._session_start = datetime.now(timezone.utc)
        self._interaction_count = 0
        self._conversation_history: list[str] = []

    def process(self, text: str, context: Optional[str] = None) -> EmotionalState:
        """Process input text and produce an emotional state.

        Traces the full pipeline: features -> detection -> situation -> state.

        Args:
            text: Input text to process.
            context: Optional contextual tag for memory.

        Returns:
            EmotionalState reflecting the system's emotional reaction.
        """
        # --- Layer 1: Feature extraction ---
        features = self.extractor.extract(text)

        # --- Layer 2: Emotion detection (keyword-based) ---
        detection = self.detector.detect(text)
        sentiment = self.analyzer.analyze(detection.emotions, text)

        # --- Layer 3: Situational analysis (event-based) ---
        situation = self.situational.analyze(text, self._conversation_history)

        # --- Layer 4: Combine signals into emotional state ---
        state = self._compute_emotional_state(
            detection.emotions, sentiment, features, situation, text
        )

        # --- Layer 5: Memory integration ---
        self.working_memory.update(state)
        self.episodic_memory.store(state, text)
        if context:
            self.semantic_memory.add_context(context, state)

        self._conversation_history.append(text)
        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-20:]
        self._interaction_count += 1

        return state

    def respond(self, text: str, context: Optional[str] = None) -> dict:
        """Process input and generate an emotionally appropriate response.

        Args:
            text: Input text to respond to.
            context: Optional contextual tag.

        Returns:
            Dict with emotional_state, response_text, and metadata.
        """
        state = self.process(text, context=context)
        response = self.response_generator.generate(state, self.personality)
        return {
            "emotional_state": state.to_dict(),
            "response_text": response,
            "sentiment": self.analyzer.analyze(state_to_emotion_scores(state)).label,
            "intensity": state.intensity,
            "primary_emotion": state.primary.value,
            "is_blended": state.blended,
        }

    def get_emotional_profile(self) -> dict:
        """Get a summary of recent emotional history."""
        recent_states = self.working_memory.get_recent(self.config.context_window)
        if not recent_states:
            return {"primary": "neutral", "intensity": 0.0}

        avg_valence = sum(s.valence for s in recent_states) / len(recent_states)
        avg_arousal = sum(s.arousal for s in recent_states) / len(recent_states)
        avg_dominance = sum(s.dominance for s in recent_states) / len(recent_states)

        return {
            "average_valence": round(avg_valence, 3),
            "average_arousal": round(avg_arousal, 3),
            "average_dominance": round(avg_dominance, 3),
            "emotional_range": self.personality.emotional_range,
            "expressiveness": self.personality.expressiveness,
            "recovery_time": self.personality.recovery_time,
            "recent_emotions": [s.primary.value for s in recent_states[-5:]],
        }

    # -----------------------------------------------------------------------
    # Internal: Emotional state computation
    # -----------------------------------------------------------------------

    def _compute_emotional_state(
        self,
        detection_scores: EmotionScores,
        sentiment,
        features,
        situation,
        text: str,
    ) -> EmotionalState:
        """Compute the final emotional state from all signal layers.

        Combines keyword detection, situational understanding, and
        context to produce a unified emotional response.
        """
        sit_ctx = situation.context

        # --- Handle temporal shifts ("but now I'm worried") ---
        # When a temporal shift is detected, re-analyze the LATER portion
        if "temporal_shift" in sit_ctx.context_tags:
            shift_match = re.search(r'\b(but\s+now|but\s+then|however|unfortunately|sadly)\b', text.lower())
            if shift_match:
                later_text = text[shift_match.end():]
                later_detection = self.detector.detect(later_text)
                later_total = sum(later_detection.emotions.emotions.values())
                if later_total > 0:
                    # The later emotion should dominate
                    detection_scores = later_detection.emotions

        # Start with keyword-detected emotions
        keyword_valence = detection_scores.valence
        keyword_arousal = detection_scores.arousal
        keyword_dominance = detection_scores.dominance
        keyword_confidence = detection_scores.confidence

        # Get situational emotional response
        sit_emotion, sit_valence, sit_arousal = _SITUATION_EMOTION_MAP.get(
            sit_ctx.situation_type, (PlutchikEmotion.TRUST, 0.0, 0.2)
        )

        # Apply severity multiplier to situational response
        severity_mult = _SEVERITY_MULTIPLIER.get(sit_ctx.severity, 1.0)
        sit_valence *= severity_mult
        sit_arousal *= severity_mult
        sit_valence = max(-1.0, min(1.0, sit_valence))
        sit_arousal = max(0.0, min(1.0, sit_arousal))

        # Shock factor: sudden events spike arousal dramatically
        shock = situation.shock_factor
        if shock > 0.5:
            sit_arousal = min(1.0, sit_arousal + shock * 0.3)

        # Combine keyword and situational signals
        # Situational signals take precedence for crisis events
        if sit_ctx.severity_score >= 0.6:
            # Crisis/high severity: situational dominates
            weight_sit = 0.7
            weight_kw = 0.3
        elif sit_ctx.severity_score >= 0.3:
            # Moderate: balanced
            weight_sit = 0.5
            weight_kw = 0.5
        else:
            # Normal: keyword dominates
            weight_sit = 0.2
            weight_kw = 0.8

        # Only blend if both signals exist
        if keyword_confidence > 0:
            combined_valence = keyword_valence * weight_kw + sit_valence * weight_sit
            combined_arousal = keyword_arousal * weight_kw + sit_arousal * weight_sit
            combined_dominance = keyword_dominance * weight_kw + 0.0 * weight_sit
        else:
            # No keywords matched -- rely on situational
            combined_valence = sit_valence
            combined_arousal = sit_arousal
            combined_dominance = 0.0

        # Clamp values
        combined_valence = max(-1.0, min(1.0, combined_valence))
        combined_arousal = max(0.0, min(1.0, combined_arousal))
        combined_dominance = max(-1.0, min(1.0, combined_dominance))

        # Determine primary emotion category
        primary_category = self._determine_primary(
            detection_scores, sit_ctx, combined_valence, combined_arousal
        )

        # Compute intensity
        intensity = self._compute_intensity(
            detection_scores, features, situation
        )

        # Compute confidence
        confidence = max(keyword_confidence, sit_ctx.severity_score * 0.8)
        if situation.distress_level > 0:
            confidence = max(confidence, situation.distress_level)
        confidence = min(1.0, confidence)

        # Build triggers list
        triggers = self._extract_triggers(text, detection_scores, situation)

        # Build dimensions metadata
        dimensions = {
            "sentiment_label": sentiment.label,
            "sentiment_score": sentiment.score,
            "exclamation_ratio": features.exclamation_ratio,
            "emotional_keywords": features.emotional_keywords_count,
            "has_negation": features.has_negation,
            "has_emphasis": features.has_emphasis,
            "situation_type": sit_ctx.situation_type.value,
            "severity": sit_ctx.severity.value,
            "urgency": round(sit_ctx.urgency, 3),
            "stakes": round(sit_ctx.stakes, 3),
            "shock_factor": round(situation.shock_factor, 3),
            "is_sudden_change": sit_ctx.is_sudden_change,
            "is_escalating": sit_ctx.is_escalating,
            "is_overwhelming": sit_ctx.is_overwhelming,
        }

        state = EmotionalState(
            primary=primary_category,
            intensity=intensity,
            valence=combined_valence,
            arousal=combined_arousal,
            dominance=combined_dominance,
            confidence=confidence,
            dimensions=dimensions,
            triggers=triggers,
        )

        # Emotional blending
        if self.config.emotional_blend_enabled:
            blended = self._blend_emotions(detection_scores, sit_ctx)
            state.blended = len(blended) > 1
            state.blended_components = blended

        return state

    def _determine_primary(
        self,
        keyword_scores: EmotionScores,
        sit_ctx,
        valence: float,
        arousal: float,
    ) -> EmotionCategory:
        """Determine the primary emotion from all signals.

        Uses keyword dominance, situational context, and dimensional position.
        """
        # Map PlutchikEmotion to EmotionCategory
        mapping = {
            PlutchikEmotion.JOY: EmotionCategory.JOY,
            PlutchikEmotion.TRUST: EmotionCategory.TRUST,
            PlutchikEmotion.FEAR: EmotionCategory.FEAR,
            PlutchikEmotion.SURPRISE: EmotionCategory.SURPRISE,
            PlutchikEmotion.SADNESS: EmotionCategory.SADNESS,
            PlutchikEmotion.DISGUST: EmotionCategory.DISGUST,
            PlutchikEmotion.ANGER: EmotionCategory.ANGER,
            PlutchikEmotion.ANTICIPATION: EmotionCategory.ANTICIPATION,
        }

        # If keywords detected something strong, use it
        total_keyword = sum(keyword_scores.emotions.values())
        if total_keyword > 0.3:
            dominant = keyword_scores.dominant_emotion()
            return mapping.get(dominant, EmotionCategory.SADNESS)

        # Use situational context
        sit_emotion, _, _ = _SITUATION_EMOTION_MAP.get(
            sit_ctx.situation_type, (PlutchikEmotion.TRUST, 0.0, 0.2)
        )

        # If crisis and keywords agree, amplify
        if sit_ctx.severity_score >= 0.6 and total_keyword > 0:
            dominant = keyword_scores.dominant_emotion()
            # Crisis + negative keyword = use keyword emotion
            if keyword_scores.valence < 0:
                return mapping.get(dominant, EmotionCategory.FEAR)
            return mapping.get(sit_emotion, EmotionCategory.FEAR)

        # Pure situational response
        if sit_ctx.situation_type != SituationType.NEUTRAL:
            return mapping.get(sit_emotion, EmotionCategory.SADNESS)

        # Dimensional fallback
        if valence > 0.3 and arousal > 0.5:
            return EmotionCategory.JOY
        elif valence < -0.3 and arousal > 0.5:
            return EmotionCategory.ANGER
        elif valence < -0.3 and arousal <= 0.5:
            return EmotionCategory.SADNESS
        elif valence > -0.3 and valence < 0.3 and arousal > 0.5:
            return EmotionCategory.FEAR
        elif valence > -0.3 and valence < 0.3 and arousal <= 0.3:
            return EmotionCategory.TRUST

        # Absolute fallback: NEVER default to joy
        if valence < 0:
            return EmotionCategory.SADNESS
        return EmotionCategory.TRUST

    def _compute_intensity(
        self,
        scores: EmotionScores,
        features,
        situation,
    ) -> float:
        """Compute emotional intensity from all signals."""
        base = max(scores.emotions.values()) if scores.emotions else 0.0

        # Emphasis boosts
        emphasis_boost = 0.0
        if features.has_emphasis:
            emphasis_boost = 0.15
        if features.exclamation_count > 0:
            emphasis_boost += 0.05 * min(features.exclamation_count, 3)

        # Situation severity boost
        severity_boost = situation.context.severity_score * 0.3

        # Shock boost
        shock_boost = situation.shock_factor * 0.2

        # Distress boost
        distress_boost = situation.distress_level * 0.3

        total = base + emphasis_boost + severity_boost + shock_boost + distress_boost
        return min(1.0, total)

    def _extract_triggers(
        self, text: str, scores: EmotionScores, situation
    ) -> list[str]:
        """Extract emotional trigger words/phrases from the text."""
        triggers: list[str] = []

        # Add matched crisis keywords
        for match, sit_type, severity in situation.raw_matches:
            triggers.append(match)

        # Add top emotion keywords
        top_emotions = scores.top_emotions(n=3)
        for emo, score in top_emotions:
            if score > 0:
                triggers.append(emo.value)

        # Add distress signals
        text_lower = text.lower()
        for signal, weight in [
            ("help", 0.3), ("scared", 0.5), ("panic", 0.5),
            ("terrible", 0.4), ("awful", 0.4), ("worst", 0.5),
            ("exhausted", 0.3), ("devastated", 0.5), ("hopeless", 0.5),
        ]:
            if signal in text_lower:
                triggers.append(signal)

        return list(dict.fromkeys(triggers))[:10]  # Deduplicate, cap at 10

    def _blend_emotions(
        self, scores: EmotionScores, sit_ctx
    ) -> list[tuple[EmotionCategory, float]]:
        """Compute blended emotions from keyword and situational signals."""
        blends: list[tuple[EmotionCategory, float]] = []

        # Keyword-based blends
        sorted_emotions = sorted(
            scores.emotions.items(), key=lambda x: x[1], reverse=True
        )
        top = [e for e, s in sorted_emotions if s > self.config.min_confidence][:3]
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                e1 = top[i]
                e2 = top[j]
                blended = self.plutchik_model.blend(e1, e2)
                if blended is not None:
                    blend_intensity = min(scores.emotions[e1], scores.emotions[e2]) * 0.7
                    cat = self._emo_to_cat(blended)
                    blends.append((cat, blend_intensity))

        # If crisis, add fear as a blend component
        if sit_ctx.severity_score >= 0.6:
            blends.append((EmotionCategory.FEAR, sit_ctx.severity_score * 0.5))

        return blends

    @staticmethod
    def _emo_to_cat(emotion: PlutchikEmotion) -> EmotionCategory:
        """Convert PlutchikEmotion to EmotionCategory."""
        mapping = {
            PlutchikEmotion.JOY: EmotionCategory.JOY,
            PlutchikEmotion.TRUST: EmotionCategory.TRUST,
            PlutchikEmotion.FEAR: EmotionCategory.FEAR,
            PlutchikEmotion.SURPRISE: EmotionCategory.SURPRISE,
            PlutchikEmotion.SADNESS: EmotionCategory.SADNESS,
            PlutchikEmotion.DISGUST: EmotionCategory.DISGUST,
            PlutchikEmotion.ANGER: EmotionCategory.ANGER,
            PlutchikEmotion.ANTICIPATION: EmotionCategory.ANTICIPATION,
        }
        return mapping.get(emotion, EmotionCategory.SADNESS)


def state_to_emotion_scores(state: EmotionalState) -> EmotionScores:
    """Convert an EmotionalState back to EmotionScores for analysis."""
    scores = EmotionScores()
    scores.valence = state.valence
    scores.arousal = state.arousal
    scores.dominance = state.dominance
    primary_map = {
        EmotionCategory.JOY: PlutchikEmotion.JOY,
        EmotionCategory.TRUST: PlutchikEmotion.TRUST,
        EmotionCategory.FEAR: PlutchikEmotion.FEAR,
        EmotionCategory.SURPRISE: PlutchikEmotion.SURPRISE,
        EmotionCategory.SADNESS: PlutchikEmotion.SADNESS,
        EmotionCategory.DISGUST: PlutchikEmotion.DISGUST,
        EmotionCategory.ANGER: PlutchikEmotion.ANGER,
        EmotionCategory.ANTICIPATION: PlutchikEmotion.ANTICIPATION,
    }
    primary_emotion = primary_map.get(state.primary)
    if primary_emotion:
        scores.emotions[primary_emotion] = state.intensity
    return scores
