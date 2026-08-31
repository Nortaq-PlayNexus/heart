"""Emotion detection via lexicon-based analysis with contextual scoring.

Detects emotions using keyword matching, phrase detection, negation handling,
intensifier modulation, and dimensional scoring. Replaces simple keyword
matching with a multi-signal approach that considers phrase patterns,
negation context, and intensity modifiers.
"""

import re
from dataclasses import dataclass, field
from heart.models.emotion_model import (
    PlutchikEmotion,
    EmotionScores,
    DIMENSIONAL_REGIONS,
)
from heart.core.state import EmotionCategory


# ---------------------------------------------------------------------------
# Expanded emotion lexicons – far more comprehensive than before
# ---------------------------------------------------------------------------

EMOTION_KEYWORDS: dict[str, list[str]] = {
    "joy": [
        "happy", "glad", "pleased", "delighted", "ecstatic", "thrilled",
        "elated", "jubilant", "cheerful", "excited", "wonderful", "amazing",
        "love", "grateful", "thankful", "blessed", "euphoric", "overjoyed",
        "content", "satisfied", "blissful", "radiant", "smile", "laugh",
        "fun", "enjoy", "beautiful", "awesome", "fantastic", "great",
        "fantastic", "perfect", "brilliant", "magnificent", "superb",
        "terrific", "splendid", "marvelous", "fabulous", "glorious",
        "joyful", "cheerful", "merry", "vibrant", "bright", "warm",
        "kind", "nice", "good", "fine", "well", "better", "best",
        "exciting", "entertaining", "pleasant", "enchanting", "charming",
        "delight", "happiness", "joy", "glory", "triumph", "celebrate",
        "proud", "impressed", "gratified", "uplifted", "inspired",
        "relief", "relieved", "comfortable", "peaceful", "calm",
        "hopeful", "optimistic", "confident", "encouraged", "motivated",
    ],
    "trust": [
        "trust", "safe", "secure", "confident", "reliable", "honest",
        "believe", "faith", "hope", "optimistic", "certain", "depend",
        "loyal", "warm", "comfort", "support", "kind", "care", "empathy",
        "sincere", "genuine", "authentic", "faithful", "devoted",
        "committed", "dedicated", "steadfast", "fearless", "brave",
        "courageous", "valiant", "bold", "assured", "guaranteed",
        "welcome", "accepted", "embraced", "valued", "appreciated",
        "respected", "honored", "admired", "cherished", "treasured",
        "bond", "connection", "intimacy", "closeness", "rapport",
        "harmony", "unity", "together", "solidarity", "partnership",
        "collaboration", "cooperation", "teamwork", "alliance",
    ],
    "fear": [
        "afraid", "scared", "anxious", "worried", "terrified", "panic",
        "dread", "nervous", "fear", "horror", "threat", "danger", "unsafe",
        "insecure", "shaky", "tremble", "frightened", "uneasy",
        "apprehensive", "stress", "stressed", "alarmed", "startled",
        "paranoid", "suspicious", "wary", "cautious", "concerned",
        "fretful", "distressed", "troubled", "bothered", "agitated",
        "restless", "edgy", "tense", "jittery", "frantic",
        "hysterical", "petrified", "spooked", "unnerved",
        "daunted", "intimidated", "overwhelmed", "threatened",
        "vulnerable", "exposed", "helpless", "powerless", "weak",
        "timid", "cowardly", "fearful", "alarmed",
        "panicking", "panicked",
    ],
    "surprise": [
        "surprise", "unexpected", "shocked", "astounded", "amazed",
        "astonished", "stunned", "wow", "incredible", "unbelievable",
        "remarkable", "novel", "strange", "odd", "whoa", "startling",
        "dramatic", "astonishing", "breathtaking", "mind-blowing",
        "unprecedented", "unforeseen", "unanticipated", "startled",
        "taken aback", "caught off guard", "jaw-dropping", "staggering",
        "stunning", "spectacular", "extraordinary", "phenomenal",
        "bewildering", "confusing", "puzzling", "mysterious", "curious",
    ],
    "sadness": [
        "sad", "sorrow", "grief", "depressed", "miserable", "heartbroken",
        "lonely", "despair", "hopeless", "gloomy", "melancholy", "crying",
        "tearful", "devastated", "mourn", "loss", "pain", "hurt", "sorry",
        "upset", "down", "low", "empty", "numb", "disappointed",
        "regret", "regretful", "remorseful", "guilty", "shame",
        "ashamed", "worthless", "defeated", "resigned", "weary",
        "exhausted", "drained", "tired", "fatigued", "worn out",
        "burned out", "depleted", "spent", "faded", "diminished",
        "lost", "miss", "missing", "yearning", "longing", "pining",
        "wistful", "nostalgic", "homesick", "heartache", "anguish",
        "agony", "suffering", "torment", "torture", "misery",
        "desolate", "forlorn", "abandoned", "neglected", "ignored",
        "forgotten", "rejected", "unwanted", "unloved", "isolated",
        "secluded", "withdrawn", "retreated", "collapsed", "crumbled",
        "broken", "shattered", "wrecked", "ruined", "destroyed",
        "devastated", "crushed", "smashed", "dismayed", "disheartened",
        "discouraged", "dispirited", "disgruntled", "dismal", "dreary",
        "bleak", "somber", "dark", "shadowy", "murky", "cloudy",
    ],
    "disgust": [
        "disgust", "revolted", "repulsive", "horrible", "icky",
        "nauseating", "offensive", "disturbing", "gross", "sick",
        "dislike", "hate", "repellent", "vile", "bad", "ugly", "worst",
        "hateful", "loathsome", "detestable", "abhorrent", "obnoxious",
        "nasty", "foul", "putrid", "rotten", "corrupt", "toxic",
        "poisonous", "contaminated", "polluted", "tainted", "spoiled",
        "ruined", "mangled", "deformed", "hideous", "grotesque",
        "ghastly", "gruesome", "macabre", "sickening", "appalling",
        "outrageous", "shameful", "disgraceful", "deplorable",
        "despicable", "contemptible", "lousy", "pathetic", "terrible",
        "dreadful", "awful", "abysmal", "atrocious", "execrable",
        "abominable", "wretched", "miserable", "horrid", "vile",
    ],
    "anger": [
        "angry", "furious", "rage", "outraged", "irritated", "annoyed",
        "mad", "livid", "enraged", "frustrated", "hostile", "resentful",
        "bitter", "aggressive", "violent", "indignant", "infuriated",
        "provoked", "offended", "displeased", "resent", "wrathful",
        "irate", "seething", "boiling", "fuming", "raging",
        "thundering", "exploding", "erupting", "burning", "scorching",
        "heated", "fiery", "passionate", "intense", "fierce",
        "savage", "ferocious", "brutal", "ruthless", "merciless",
        "punishing", "attacking", "assaulting", "fighting", "warring",
        "conflict", "battle", "struggle", "rebel", "rebellion",
        "defiance", "defiant", "rebellious", "obstinate", "stubborn",
        "uncooperative", "difficult", "problem", "trouble", "bother",
        "nuisance", "pest", "irritation", "aggravation", "exasperation",
        "panicking", "panicked",
    ],
    "anticipation": [
        "anticipate", "expect", "look forward", "eager", "excited",
        "prepared", "ready", "impatient", "curious", "wonder", "await",
        "planning", "future", "promising", "thrilled", "exciting",
        "awaiting", "pending", "approaching", "coming", "upcoming",
        "imminent", "looming", "emerging", "developing", "evolving",
        "progressing", "advancing", "moving", "driving", "pushing",
        "driven", "ambitious", "aspiring", "striving", "seeking",
        "pursuing", "chasing", "hunting", "searching", "exploring",
        "discovering", "inventing", "creating", "building", "designing",
        "dreaming", "imagining", "envisioning", "conceiving", "planning",
        "preparing", "organizing", "arranging", "scheduling", "booking",
    ],
}

# ---------------------------------------------------------------------------
# Compound phrases – detected as units before individual words
# ---------------------------------------------------------------------------

PHRASE_EMOTIONS: dict[str, tuple[str, float]] = {
    # phrase -> (emotion, base_score)
    "bad day": ("sadness", 0.7),
    "terrible day": ("sadness", 0.8),
    "awful day": ("sadness", 0.7),
    "great day": ("joy", 0.8),
    "wonderful day": ("joy", 0.8),
    "feel terrible": ("sadness", 0.8),
    "feel awful": ("sadness", 0.8),
    "feel exhausted": ("sadness", 0.7),
    "feel drained": ("sadness", 0.7),
    "feel tired": ("sadness", 0.5),
    "feel sick": ("disgust", 0.7),
    "feel angry": ("anger", 0.8),
    "feel happy": ("joy", 0.8),
    "feel scared": ("fear", 0.8),
    "feel afraid": ("fear", 0.8),
    "feel anxious": ("fear", 0.7),
    "feel sad": ("sadness", 0.8),
    "feel depressed": ("sadness", 0.9),
    "feel lonely": ("sadness", 0.8),
    "feel hopeless": ("sadness", 0.9),
    "feel lost": ("sadness", 0.7),
    "feel empty": ("sadness", 0.8),
    "feel numb": ("sadness", 0.7),
    "feel broken": ("sadness", 0.8),
    "feel worthless": ("sadness", 0.9),
    "feel stupid": ("sadness", 0.7),
    "feel useless": ("sadness", 0.8),
    "feel like crying": ("sadness", 0.9),
    "can't stop crying": ("sadness", 0.95),
    "worst day": ("sadness", 0.9),
    "best day": ("joy", 0.9),
    "worse than": ("sadness", 0.6),
    "hate this": ("anger", 0.7),
    "hate it": ("anger", 0.7),
    "makes me sick": ("disgust", 0.8),
    "sick of": ("disgust", 0.7),
    "fed up": ("anger", 0.7),
    "had enough": ("anger", 0.7),
    "can't take": ("anger", 0.7),
    "drives me crazy": ("anger", 0.8),
    "pissed off": ("anger", 0.8),
    "blood boil": ("anger", 0.8),
    "makes my blood boil": ("anger", 0.9),
    "stressed out": ("fear", 0.7),
    "freaking out": ("fear", 0.8),
    "falling apart": ("sadness", 0.9),
    "giving up": ("sadness", 0.9),
    "nothing matters": ("sadness", 0.9),
    "no point": ("sadness", 0.8),
    "what's the point": ("sadness", 0.8),
    "end it all": ("sadness", 0.95),
    "want to die": ("sadness", 0.95),
    "kill myself": ("sadness", 0.95),
    "better off dead": ("sadness", 0.95),
    "hurt myself": ("sadness", 0.9),
    "miss you": ("sadness", 0.7),
    "miss them": ("sadness", 0.7),
    "so alone": ("sadness", 0.8),
    "all alone": ("sadness", 0.8),
    "no one cares": ("sadness", 0.9),
    "nobody cares": ("sadness", 0.9),
    "never happy": ("sadness", 0.8),
    "always sad": ("sadness", 0.8),
    "feel nothing": ("sadness", 0.8),
    "don't care": ("sadness", 0.6),
    "can't sleep": ("fear", 0.6),
    "nightmare": ("fear", 0.7),
    "scared to death": ("fear", 0.9),
    "terrified of": ("fear", 0.9),
    "worried about": ("fear", 0.7),
    "anxious about": ("fear", 0.7),
    "stressed about": ("fear", 0.7),
}

# ---------------------------------------------------------------------------
# Negation words and their scope rules
# ---------------------------------------------------------------------------

NEGATION_WORDS = {
    "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
    "nor", "cannot", "can't", "won't", "wouldn't", "shouldn't", "couldn't",
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't", "shan't", "may not", "might not",
    "barely", "hardly", "scarcely", "without",
}

# Words that negate the NEXT emotion word within a window
NEGATION_TRIGGERS = {
    "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
    "nor", "cannot", "don't", "doesn't", "didn't", "isn't", "aren't",
    "wasn't", "weren't", "hasn't", "haven't", "hadn't", "won't",
    "wouldn't", "shouldn't", "couldn't", "can't", "shan't",
}

# Exceptions: phrases where negation doesn't actually negate emotion
NON_NEGATING_PHRASES = {
    "can't believe",  # "I can't believe how great" = positive
    "can't wait",     # "I can't wait" = positive anticipation
    "no way",         # "No way!" = surprise/joy
    "not bad",        # "Not bad" = mildly positive
    "not only",       # "Not only... but also" = emphasis
}

# ---------------------------------------------------------------------------
# Intensifiers and diminishers
# ---------------------------------------------------------------------------

INTENSIFIERS: dict[str, float] = {
    "very": 1.4,
    "extremely": 1.8,
    "incredibly": 1.7,
    "absolutely": 1.6,
    "totally": 1.5,
    "completely": 1.5,
    "utterly": 1.6,
    "deeply": 1.4,
    "overwhelmingly": 1.8,
    "immensely": 1.7,
    "terribly": 1.4,
    "awfully": 1.4,
    "dreadfully": 1.4,
    "really": 1.3,
    "so": 1.3,
    "too": 1.3,
    "quite": 1.2,
    "rather": 1.1,
    "somewhat": 0.7,
    "slightly": 0.5,
    "barely": 0.3,
    "hardly": 0.3,
    "a bit": 0.6,
    "a little": 0.6,
    "kind of": 0.5,
    "sort of": 0.5,
}


@dataclass
class DetectionResult:
    """Result of emotion detection on a text input."""
    emotions: EmotionScores
    text_processed: str
    confidence: float = 0.0
    word_count: int = 0
    negation_detected: bool = False
    intensifier_factor: float = 1.0
    matched_phrases: list[str] = field(default_factory=list)
    matched_keywords: dict[str, list[str]] = field(default_factory=dict)


class EmotionDetector:
    """Detects emotions in text using lexicon matching, phrase detection,
    negation handling, and intensifier modulation.

    Uses a multi-signal approach:
    1. Compound phrase detection (highest priority)
    2. Individual keyword matching with context window
    3. Negation reversal within scope
    4. Intensifier/diminisher modulation
    5. Dimensional scoring aggregation
    """

    def __init__(self):
        self._compiled_patterns: dict[str, re.Pattern] = {}
        for emotion, keywords in EMOTION_KEYWORDS.items():
            pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
            self._compiled_patterns[emotion] = re.compile(pattern, re.IGNORECASE)

        self._phrase_patterns: list[tuple[re.Pattern, str, float]] = []
        for phrase, (emotion, score) in PHRASE_EMOTIONS.items():
            pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
            self._phrase_patterns.append((pattern, emotion, score))

    def detect(self, text: str) -> DetectionResult:
        """Detect emotions in the given text.

        Args:
            text: Input text to analyze.

        Returns:
            DetectionResult with emotion scores, confidence, and metadata.
        """
        text_lower = text.lower()
        scores = EmotionScores()
        word_count = len(text.split())
        matched_phrases: list[str] = []
        matched_keywords: dict[str, list[str]] = {}

        # --- Phase 1: Negation detection (contextual) ---
        negation_zones = self._find_negation_zones(text_lower)

        # --- Phase 2: Intensifier detection ---
        intensifier_factor = 1.0
        for intensifier, factor in INTENSIFIERS.items():
            if intensifier in text_lower:
                intensifier_factor = max(intensifier_factor, factor)

        # --- Phase 3: Compound phrase detection ---
        phrase_scores: dict[str, float] = {}
        for pattern, emotion, base_score in self._phrase_patterns:
            matches = pattern.findall(text_lower)
            if matches:
                for match in matches:
                    phrase_scores[emotion] = max(phrase_scores.get(emotion, 0), base_score)
                matched_phrases.extend(matches)

        for emotion, score in phrase_scores.items():
            emotion_enum = PlutchikEmotion(emotion)
            scores.emotions[emotion_enum] = max(scores.emotions.get(emotion_enum, 0), score)

        # --- Phase 4: Individual keyword detection with context ---
        for emotion_name, pattern in self._compiled_patterns.items():
            matches = pattern.findall(text_lower)
            if not matches:
                continue

            emotion_enum = PlutchikEmotion(emotion_name)
            matched_keywords[emotion_name] = matches

            # Count non-negated matches
            positive_count = 0
            negated_count = 0
            for match in matches:
                match_start = text_lower.find(match)
                if self._is_negated(match_start, negation_zones):
                    negated_count += 1
                else:
                    positive_count += 1

            if positive_count > 0:
                base_score = min(positive_count / max(word_count * 0.1, 1), 1.0)
                base_score *= intensifier_factor
                # Boost if phrase also matched this emotion
                if emotion_name in phrase_scores:
                    base_score = max(base_score, phrase_scores[emotion_name])
                scores.emotions[emotion_enum] = max(
                    scores.emotions.get(emotion_enum, 0), base_score
                )

            if negated_count > 0:
                # Negated emotion: add score to the OPPOSITE emotion
                from heart.models.emotion_model import EMOTION_OPPOSITES
                opposite = EMOTION_OPPOSITES.get(emotion_enum)
                if opposite:
                    neg_score = min(negated_count / max(word_count * 0.1, 1), 0.8)
                    neg_score *= intensifier_factor
                    scores.emotions[opposite] = max(
                        scores.emotions.get(opposite, 0), neg_score
                    )

        # --- Phase 5: Dimensional scoring ---
        total = sum(scores.emotions.values())
        if total > 0:
            for emotion_name, region in DIMENSIONAL_REGIONS.items():
                v_range = region["valence"]
                a_range = region["arousal"]
                emotion_enum = PlutchikEmotion(emotion_name)
                current = scores.emotions.get(emotion_enum, 0.0)
                if current > 0.0:
                    target_valence = (v_range[0] + v_range[1]) / 2
                    target_arousal = (a_range[0] + a_range[1]) / 2
                    scores.valence += current * target_valence
                    scores.arousal += current * target_arousal
                    scores.dominance += current * (0.0 + current * 0.5)

            scores.valence /= total
            scores.arousal /= total
            scores.dominance /= total
            scores.valence = max(-1.0, min(1.0, scores.valence))
            scores.arousal = max(0.0, min(1.0, scores.arousal))
            scores.dominance = max(-1.0, min(1.0, scores.dominance))
            scores.confidence = min(total / max(word_count * 0.05, 1), 1.0)
        else:
            # No emotions detected – use a heuristic from the text itself
            scores = self._heuristic_fallback(text_lower, scores)

        negation_detected = len(negation_zones) > 0

        return DetectionResult(
            emotions=scores,
            text_processed=text_lower,
            confidence=scores.confidence,
            word_count=word_count,
            negation_detected=negation_detected,
            intensifier_factor=intensifier_factor,
            matched_phrases=matched_phrases,
            matched_keywords=matched_keywords,
        )

    def _find_negation_zones(self, text_lower: str) -> list[tuple[int, int]]:
        """Find character ranges where negation applies.

        Negation applies from the negation word up to the next punctuation
        or clause boundary (up to 5 words ahead). Excludes non-negating
        phrases like "can't believe" and "can't wait".
        """
        zones: list[tuple[int, int]] = []
        words = text_lower.split()
        char_pos = 0
        for i, word in enumerate(words):
            clean = word.strip(".,!?;:()\"'")
            if clean in NEGATION_TRIGGERS:
                # Check for non-negating phrases
                phrase_context = " ".join(words[i:i + 3])
                is_non_negating = any(phrase in phrase_context for phrase in NON_NEGATING_PHRASES)
                if is_non_negating:
                    char_pos += len(word) + 1
                    continue

                start = char_pos
                # Negation scope: up to 5 words or next punctuation
                remaining = " ".join(words[i + 1:i + 6])
                end = start + len(remaining) + len(word) + 1
                zones.append((start, min(end, len(text_lower))))
            char_pos += len(word) + 1
        return zones

    def _is_negated(self, position: int, zones: list[tuple[int, int]]) -> bool:
        """Check if a character position falls within a negation zone."""
        for start, end in zones:
            if start <= position <= end:
                return True
        return False

    def _heuristic_fallback(
        self, text_lower: str, scores: EmotionScores
    ) -> EmotionScores:
        """When no keywords match, use basic heuristics to infer emotion.

        Checks for common negative patterns that the lexicon may miss.
        """
        negative_patterns = [
            (r'\bbad\b', "sadness", 0.5),
            (r'\bbad\b.*\bday\b', "sadness", 0.7),
            (r'\bterrible\b', "disgust", 0.7),
            (r'\bawful\b', "disgust", 0.7),
            (r'\bworst\b', "disgust", 0.8),
            (r'\bhorrible\b', "disgust", 0.7),
            (r'\bexhausted\b', "sadness", 0.6),
            (r'\btired\b', "sadness", 0.4),
            (r'\bsick\b', "disgust", 0.6),
            (r'\bhate\b', "anger", 0.7),
            (r'\bkill\b', "anger", 0.6),
            (r'\bdie\b', "sadness", 0.7),
            (r'\bdeath\b', "sadness", 0.7),
            (r'\bdead\b', "sadness", 0.7),
            (r'\bcried\b', "sadness", 0.8),
            (r'\bcrying\b', "sadness", 0.8),
            (r'\bsuffering\b', "sadness", 0.7),
            (r'\bworse\b', "sadness", 0.6),
            (r'\bworst\b', "disgust", 0.8),
            (r'\bupset\b', "sadness", 0.6),
            (r'\bfrustrated\b', "anger", 0.7),
            (r'\bannoyed\b', "anger", 0.6),
            (r'\bfurious\b', "anger", 0.9),
            (r'\brage\b', "anger", 0.9),
            (r'\bpanicked\b', "fear", 0.8),
            (r'\bworthless\b', "sadness", 0.8),
            (r'\buseless\b', "sadness", 0.7),
            (r'\bpathetic\b', "disgust", 0.7),
            (r'\bmiserable\b', "sadness", 0.8),
            (r'\bdepressed\b', "sadness", 0.9),
            (r'\blonely\b', "sadness", 0.8),
            (r'\bempty\b', "sadness", 0.6),
            (r'\bnumb\b', "sadness", 0.6),
            (r'\bhopeless\b', "sadness", 0.9),
            (r'\bgloomy\b', "sadness", 0.7),
            (r'\bbleak\b', "sadness", 0.7),
            (r'\bdreadful\b', "disgust", 0.7),
            (r'\brotten\b', "disgust", 0.7),
            (r'\bfoul\b', "disgust", 0.7),
            (r'\bnasty\b', "disgust", 0.7),
        ]

        best_emotion = None
        best_score = 0.0
        for pattern_str, emotion, base_score in negative_patterns:
            if re.search(pattern_str, text_lower):
                if base_score > best_score:
                    best_score = base_score
                    best_emotion = emotion

        if best_emotion and best_score > 0:
            emotion_enum = PlutchikEmotion(best_emotion)
            scores.emotions[emotion_enum] = best_score
            # Apply dimensional scoring
            region = DIMENSIONAL_REGIONS.get(best_emotion, {})
            if region:
                v_range = region.get("valence", (0.0, 0.0))
                a_range = region.get("arousal", (0.0, 0.0))
                scores.valence = (v_range[0] + v_range[1]) / 2
                scores.arousal = (a_range[0] + a_range[1]) / 2
                scores.dominance = 0.0
            scores.confidence = best_score * 0.7

        return scores


def classify_emotion(scores: EmotionScores) -> EmotionCategory:
    """Map the dominant EmotionScores emotion to an EmotionCategory."""
    dominant = scores.dominant_emotion()
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
    return mapping.get(dominant, EmotionCategory.SADNESS)
