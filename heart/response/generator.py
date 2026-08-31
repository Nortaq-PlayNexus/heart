"""Emotionally intelligent response generator with situational awareness.

Produces responses that feel like genuine reactions to events, not just
emotionally-flavored templates. Handles crises, jumpscares, overwhelm,
and emotional momentum.
"""

import random
from heart.core.state import EmotionalState, EmotionCategory
from heart.models.personality import PersonalityProfile


class ResponseGenerator:
    """Generates emotionally appropriate, situationally-aware responses.

    Produces responses that reflect genuine emotional reactions:
    - Crisis responses convey urgency and alarm
    - Jumpscare responses convey shock and being caught off guard
    - Overwhelm responses convey being flooded
    - Normal responses match the detected emotional tone
    """

    def __init__(self):
        self._templates: dict[EmotionCategory, list[str]] = {
            EmotionCategory.JOY: [
                "That's wonderful! I can feel the positive energy in what you're sharing.",
                "Your joy is contagious. It's great to hear something so uplifting.",
                "I sense real happiness here. That's something worth celebrating.",
                "There's a brightness in what you're describing that really stands out.",
                "I can feel the warmth and positivity you're expressing.",
                "That genuinely makes me happy to hear. What a great moment.",
                "There's something really beautiful about what you're describing.",
            ],
            EmotionCategory.SADNESS: [
                "I hear the heaviness in your words. It's okay to feel this way.",
                "What you're going through sounds difficult. Your feelings are valid.",
                "I sense real sadness in what you're sharing. Would you like to talk about it?",
                "Sometimes things feel overwhelming. I'm here to listen.",
                "The weight of this situation comes through clearly. Take the time you need.",
                "That sounds genuinely painful. I'm sorry you're going through this.",
                "I can feel the sadness in what you're describing. You don't have to face it alone.",
            ],
            EmotionCategory.ANGER: [
                "I can sense the frustration in your message. That's completely understandable.",
                "It sounds like something has really pushed your buttons.",
                "Your anger is valid given the situation you're describing.",
                "I understand this is frustrating. Let's think through it together.",
                "That does sound infuriating. Anyone would feel this way in your position.",
                "I feel the frustration too. That's a genuinely aggravating situation.",
            ],
            EmotionCategory.FEAR: [
                "I sense real anxiety in what you're sharing. That's a natural response.",
                "It's completely understandable to feel uncertain about this. Let's work through it.",
                "I can feel the apprehension in your words. You're not alone in this.",
                "That's a scary situation. Your concern makes total sense.",
                "I understand the worry here. Let's figure out what we can do.",
                "That would make anyone nervous. Your fear is completely warranted.",
            ],
            EmotionCategory.SURPRISE: [
                "That does sound unexpected! How are you processing it?",
                "I can feel the surprise in what you're describing. That's quite a moment.",
                "Unexpected situations can be jarring. Take a moment to absorb it all.",
                "That's quite a twist! I understand why you'd feel caught off guard.",
                "Sometimes life throws curveballs. Your reaction makes complete sense.",
            ],
            EmotionCategory.DISGUST: [
                "I understand that reaction. Some things are genuinely off-putting.",
                "Your response makes sense given what you encountered.",
                "That does sound unpleasant. Trusting your instincts is wise.",
                "I can sense the aversion in your words. Your boundaries matter.",
                "It's natural to feel repelled by something that crosses a line.",
            ],
            EmotionCategory.TRUST: [
                "It sounds like you're building a meaningful connection.",
                "Trust is a beautiful foundation. I can feel the security in your words.",
                "There's a sense of confidence and safety in what you're expressing.",
                "The trust you're describing is really valuable. Guard it well.",
                "I sense genuine openness and faith in what you're sharing.",
            ],
            EmotionCategory.ANTICIPATION: [
                "I can feel the anticipation building in your words!",
                "You seem to be looking forward to something. That excitement is palpable.",
                "There's a wonderful sense of expectation in what you're describing.",
                "The eagerness in your message is clear. Something good is on the horizon.",
                "Anticipation like this means something matters deeply to you.",
            ],
        }

        # Crisis-specific responses -- these are REACTIONS, not templates
        self._crisis_responses: dict[str, list[str]] = {
            "data_loss": [
                "Wait -- you deleted system32?! That's... that's a serious problem. I need you to stop whatever you're doing right now.",
                "Okay, deep breath. Deleting system32 is critical. Let's figure out if we can recover from this.",
                "That's alarming. System32 is essential for Windows to function. We need to act fast here.",
                "I'm genuinely concerned. Losing system32 means the OS won't boot. Do you have a recovery option?",
            ],
            "security_breach": [
                "That's a security emergency. We need to isolate the system immediately.",
                "A breach is serious. I'm feeling the urgency here -- let's lock things down.",
                "Okay, this is critical. If there's a malware infection, we need to contain it now.",
                "I sense real danger here. Let's get your system secured before we do anything else.",
            ],
            "system_failure": [
                "A system failure is stressful. Let's work through this step by step.",
                "I can feel the frustration. Crashes are never fun, but we can troubleshoot this.",
                "That's annoying. Let me help you figure out what's causing this.",
            ],
            "emergency": [
                "This sounds urgent. Please prioritize your safety above everything else.",
                "I'm concerned. If this is a real emergency, please reach out to emergency services.",
                "Your wellbeing matters most right now. Let's focus on what's most important.",
            ],
            "escalation": [
                "I can feel things spiraling. Let's take a step back and assess what's happening.",
                "When things escalate like this, it helps to pause and breathe. What's the core issue?",
                "I sense the situation intensifying. Let's try to stabilize things first.",
            ],
            "overwhelm": [
                "That's a lot to process all at once. Let's break this down into manageable pieces.",
                "I feel the overwhelm too. When everything hits at once, it's hard to think clearly.",
                "Too much at once is genuinely stressful. Let's focus on one thing at a time.",
                "I'm getting hit with all of this too. Let's slow down and tackle it piece by piece.",
            ],
            "vulnerability": [
                "I hear you. That kind of emptiness is real and it matters.",
                "What you're feeling is valid. You don't have to pretend to be okay.",
                "I sense the pain behind those words. You're not as alone as it might feel.",
                "That takes courage to say. Your feelings matter, even the dark ones.",
            ],
            "setback": [
                "That's a tough setback. It's okay to feel frustrated about it.",
                "I understand the disappointment. Setbacks feel awful, but they're not the end.",
                "That didn't go as planned. Let's figure out what we can learn from this.",
            ],
            "breakthrough": [
                "That's a real breakthrough! I can feel the relief and accomplishment.",
                "You figured it out! That's genuinely exciting.",
                "Nice work! That moment when things finally click is the best.",
            ],
            "resolution": [
                "I'm glad it's resolved. That's a weight off both our shoulders.",
                "That's a relief! Everything's back on track.",
                "Good to hear things are working again.",
            ],
        }

        # Jumpscare response -- for sudden, shocking negative events
        self._jumpscare_responses = [
            "Whoa -- that took a turn. I genuinely wasn't expecting that.",
            "Okay, that's alarming. My attention is fully on this now.",
            "That's... not what I expected to hear. Let me process this for a moment.",
            "I felt that. That's a sudden shift and it's concerning.",
            "That hit me off guard. This is serious and we need to address it.",
        ]

    def generate(self, state: EmotionalState, personality: PersonalityProfile | None = None) -> str:
        """Generate an emotionally appropriate response.

        Considers the emotional state, situation type, intensity, and
        personality to produce a response that feels like a genuine reaction.

        Args:
            state: The current emotional state.
            personality: Optional personality profile for modulation.

        Returns:
            A response string appropriate to the emotional context.
        """
        dims = state.dimensions or {}
        situation_type = dims.get("situation_type", "neutral")
        severity = dims.get("severity", "normal")
        shock = dims.get("shock_factor", 0.0)
        is_overwhelming = dims.get("is_overwhelming", False)
        urgency = dims.get("urgency", 0.0)

        # --- Crisis response (highest priority) ---
        if severity in ("crisis", "catastrophic"):
            crisis_key = self._get_crisis_key(situation_type)
            if crisis_key and crisis_key in self._crisis_responses:
                response = random.choice(self._crisis_responses[crisis_key])
                if personality:
                    response = self._apply_personality(response, state, personality)
                return response

        # --- Jumpscare response (sudden shocking NEGATIVE events) ---
        # Only trigger for negative situations -- positive excitement is not a jumpscare
        # Also only trigger if keywords didn't already detect a strong emotion
        is_negative_situation = state.valence < 0 or severity in ("crisis", "catastrophic")
        keywords_detected = state.confidence > 0.3
        if is_negative_situation and not keywords_detected and (shock > 0.5 or (situation_type in ("crisis_event", "data_loss", "security_breach")
                           and state.arousal > 0.7)):
            response = random.choice(self._jumpscare_responses)
            if personality:
                response = self._apply_personality(response, state, personality)
            return response

        # --- Overwhelm response ---
        if is_overwhelming:
            response = random.choice(self._crisis_responses.get("overwhelm", []))
            if personality:
                response = self._apply_personality(response, state, personality)
            return response

        # --- Escalation response ---
        if dims.get("is_escalating", False):
            response = random.choice(self._crisis_responses.get("escalation", []))
            if personality:
                response = self._apply_personality(response, state, personality)
            return response

        # --- High urgency response ---
        if urgency > 0.7:
            templates = self._crisis_responses.get(situation_type, [])
            if templates:
                response = random.choice(templates)
                if personality:
                    response = self._apply_personality(response, state, personality)
                return response

        # --- Standard emotional response ---
        templates = self._templates.get(state.primary, [])
        if not templates:
            return "I understand. Thank you for sharing that with me."

        if state.intensity > 0.7:
            response = self._select_emphatic(templates)
        elif state.intensity < 0.3:
            response = self._select_gentle(templates)
        else:
            response = random.choice(templates)

        if personality:
            response = self._apply_personality(response, state, personality)

        # Add blend note if applicable
        if state.blended and state.blended_components:
            secondary = state.blended_components[0][0] if state.blended_components else None
            if secondary is not None:
                blend_note = self._get_blend_note(state.primary, secondary)
                if blend_note:
                    response = f"{response} {blend_note}"

        return response

    def _get_crisis_key(self, situation_type: str) -> str | None:
        """Map situation type to crisis response key."""
        mapping = {
            "data_loss": "data_loss",
            "security_breach": "security_breach",
            "system_failure": "system_failure",
            "emergency": "emergency",
            "crisis_event": "data_loss",  # General crisis uses data_loss templates
            "escalation": "escalation",
            "vulnerability": "vulnerability",
            "setback": "setback",
            "breakthrough": "breakthrough",
            "resolution": "resolution",
        }
        return mapping.get(situation_type)

    def _select_emphatic(self, templates: list[str]) -> str:
        """Select an emphatic response for high-intensity emotions."""
        emphatic_markers = ["genuine", "really", "palpable", "clear", "concern", "serious"]
        for template in templates:
            if any(marker in template.lower() for marker in emphatic_markers):
                return template
        return random.choice(templates)

    def _select_gentle(self, templates: list[str]) -> str:
        """Select a gentle response for low-intensity emotions."""
        gentle_markers = ["okay", "natural", "understand", "reasonable", "sense", "moment"]
        for template in templates:
            if any(marker in template.lower() for marker in gentle_markers):
                return template
        return random.choice(templates)

    def _apply_personality(
        self, response: str, state: EmotionalState, personality: PersonalityProfile
    ) -> str:
        """Apply personality-based modulation to a response."""
        neuroticism = personality.traits.neuroticism
        extraversion = personality.traits.extraversion
        agreeableness = personality.traits.agreeableness

        if neuroticism > 0.7 and state.valence < 0:
            prefix = random.choice([
                "I really feel for you. ",
                "This must be so hard. ",
                "I sense this deeply. ",
            ])
            response = prefix + response

        if extraversion > 0.7 and state.valence > 0:
            response = response.rstrip(".") + "!"

        if agreeableness > 0.7 and state.valence < 0:
            support = random.choice([
                " Remember, you're not alone in this.",
                " I'm here for you.",
                " We can face this together.",
            ])
            response = response + support

        return response

    def _get_blend_note(self, primary: EmotionCategory, secondary: EmotionCategory) -> str:
        """Generate a note about blended emotional states."""
        blend_notes = {
            (EmotionCategory.JOY, EmotionCategory.TRUST): "There's a warmth of trust woven through this happiness.",
            (EmotionCategory.SADNESS, EmotionCategory.ANGER): "I sense both sorrow and frustration intertwined.",
            (EmotionCategory.FEAR, EmotionCategory.SURPRISE): "There's an element of unexpected alarm here.",
            (EmotionCategory.ANGER, EmotionCategory.DISGUST): "The frustration seems tinged with something deeper.",
            (EmotionCategory.FEAR, EmotionCategory.SADNESS): "I feel both the fear and the sadness in this.",
            (EmotionCategory.SADNESS, EmotionCategory.FEAR): "There's grief mixed with anxiety here.",
        }
        pair = (primary, secondary)
        return blend_notes.get(pair, "")
