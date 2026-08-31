"""Situational understanding and crisis detection for H.E.A.R.T.

Goes beyond keyword matching to understand EVENTS and SITUATIONS.
Detects crises, urgency, severity, and contextual meaning so the
system can genuinely react to what's happening, not just what's said.
"""

import re
from dataclasses import dataclass, field
from enum import Enum


class SeverityLevel(Enum):
    """Severity of a detected situation."""
    NORMAL = "normal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRISIS = "crisis"
    CATASTROPHIC = "catastrophic"


class SituationType(Enum):
    """Types of situations the system can recognize."""
    NEUTRAL = "neutral"
    TASK_IN_PROGRESS = "task_in_progress"
    PROBLEM_REPORTED = "problem_reported"
    CRISIS_EVENT = "crisis_event"
    DATA_LOSS = "data_loss"
    SECURITY_BREACH = "security_breach"
    SYSTEM_FAILURE = "system_failure"
    EMERGENCY = "emergency"
    RESOLUTION = "resolution"
    ESCALATION = "escalation"
    DEESCALATION = "deescalation"
    MISUNDERSTANDING = "misunderstanding"
    PERSONAL_SHARING = "personal_sharing"
    JOINT_CELEBRATION = "joint_celebration"
    CONFLICT = "conflict"
    VULNERABILITY = "vulnerability"
    OVERWHELM = "overwhelm"
    BREAKTHROUGH = "breakthrough"
    SETBACK = "setback"


@dataclass
class SituationalContext:
    """Understanding of the current situation."""
    situation_type: SituationType = SituationType.NEUTRAL
    severity: SeverityLevel = SeverityLevel.NORMAL
    urgency: float = 0.0           # 0.0 (no urgency) to 1.0 (immediate)
    stakes: float = 0.0            # 0.0 (trivial) to 1.0 (life-changing)
    emotional_charge: float = 0.0  # -1.0 (deeply negative) to 1.0 (deeply positive)
    is_sudden_change: bool = False  # Did the situation change dramatically?
    is_escalating: bool = False    # Is the situation getting worse?
    is_overwhelming: bool = False  # Is there too much happening at once?
    crisis_keywords: list[str] = field(default_factory=list)
    context_tags: list[str] = field(default_factory=list)
    severity_score: float = 0.0    # Numeric severity for calculations


# ---------------------------------------------------------------------------
# Crisis and situation patterns - what EVENTS mean, not just what WORDS mean
# ---------------------------------------------------------------------------

CRISIS_PATTERNS: list[tuple[str, SituationType, SeverityLevel, float, float]] = [
    # (regex_pattern, situation_type, severity, urgency, stakes)
    # --- Data loss ---
    (r"deleted?\s+(my\s+)?(system32|windows|boot|os|operating\s+system)",
     SituationType.DATA_LOSS, SeverityLevel.CATASTROPHIC, 1.0, 1.0),
    (r"(deleted?\s+(my\s+)?(files?|data|documents?|photos?|backup))|"
     r"(deleted?\s+everything)|"
     r"(lost\s+all\s+(my\s+)?(important\s+)?(files?|data|documents?|work|progress))|"
     r"(losing\s+(all\s+)?(my\s+)?(data|files?|work|progress))",
     SituationType.DATA_LOSS, SeverityLevel.HIGH, 0.8, 0.8),
    (r"formatted?\s+(my\s+)?(drive|disk|partition|drive)",
     SituationType.DATA_LOSS, SeverityLevel.CATASTROPHIC, 1.0, 1.0),
    (r"rm\s+-rf\s+/", SituationType.DATA_LOSS, SeverityLevel.CATASTROPHIC, 1.0, 1.0),
    (r"rm\s+-rf\s+~", SituationType.DATA_LOSS, SeverityLevel.CRISIS, 0.95, 0.9),
    (r"blue\s+screen|bsod|blue\s+screen\s+of\s+death",
     SituationType.SYSTEM_FAILURE, SeverityLevel.HIGH, 0.8, 0.7),
    # --- Security ---
    (r"(hacked|compromised|breached|intrusion|malware|ransomware)",
     SituationType.SECURITY_BREACH, SeverityLevel.CRISIS, 0.9, 0.9),
    (r"(has\s+a?\s*virus|infected\s+with|virus\s+scan\s+found|virus\s+detected)",
     SituationType.SECURITY_BREACH, SeverityLevel.HIGH, 0.8, 0.7),
    (r"(stolen|leaked)\s+(password|credentials?|data|info)",
     SituationType.SECURITY_BREACH, SeverityLevel.CRISIS, 0.95, 0.85),
    (r"(keylogger|trojan|rootkit|backdoor)",
     SituationType.SECURITY_BREACH, SeverityLevel.HIGH, 0.85, 0.8),
    # --- System failure ---
    (r"(crashed?|crashing?|freeze|freezing?|frozen|hang|hanging?)",
     SituationType.SYSTEM_FAILURE, SeverityLevel.MODERATE, 0.6, 0.5),
    (r"(won't\s+boot|can't\s+boot|boot\s+loop|restart\s+loop)",
     SituationType.SYSTEM_FAILURE, SeverityLevel.HIGH, 0.85, 0.8),
    (r"(kernel\s+panic|fatal\s+error|critical\s+error)",
     SituationType.SYSTEM_FAILURE, SeverityLevel.CRISIS, 0.9, 0.85),
    # --- Emergency ---
    (r"(call\s+911|ambulance|hospital|emergency|dying|heart\s+attack)",
     SituationType.EMERGENCY, SeverityLevel.CATASTROPHIC, 1.0, 1.0),
    (r"(suicide|kill\s+myself|end\s+it\s+all|want\s+to\s+die)",
     SituationType.EMERGENCY, SeverityLevel.CATASTROPHIC, 1.0, 1.0),
    # --- Escalation ---
    (r"(getting\s+worse|escalating|spinning\s+out\s+of\s+control|falling\s+apart|"
     r"spiraling\s+(out\s+of\s+control|downwards|out\s+of\s+hand))",
     SituationType.ESCALATION, SeverityLevel.HIGH, 0.8, 0.7),
    (r"(everything\s+is\s+wrong|nothing\s+works|total\s+failure)",
     SituationType.ESCALATION, SeverityLevel.HIGH, 0.75, 0.7),
    # --- Resolution ---
    (r"(fixed\s+it|solved|working\s+again|problem\s+solved|all\s+good)",
     SituationType.RESOLUTION, SeverityLevel.LOW, 0.1, 0.1),
    (r"(recovered|restored|backup\s+worked|got\s+it\s+back)",
     SituationType.RESOLUTION, SeverityLevel.LOW, 0.1, 0.2),
    # --- Overwhelm ---
    (r"(too\s+much|overwhelming|overwhelmed|drowning|flooded|spammed)",
     SituationType.OVERWHELM, SeverityLevel.HIGH, 0.7, 0.6),
    (r"(wall\s+of\s+text|code\s+everywhere|error\s+messages?\s+everywhere)",
     SituationType.OVERWHELM, SeverityLevel.MODERATE, 0.6, 0.5),
    (r"(error\s+error\s+error|exception\s+traceback|stack\s+trace)",
     SituationType.OVERWHELM, SeverityLevel.MODERATE, 0.5, 0.4),
    # --- Personal vulnerability ---
    (r"(i\s+(feel|am)\s+(lonely|alone|empty|numb|worthless|useless|broken))",
     SituationType.VULNERABILITY, SeverityLevel.HIGH, 0.5, 0.7),
    (r"(nobody\s+(cares?|loves?\s+me)|no\s+one\s+(cares?|is\s+there))",
     SituationType.VULNERABILITY, SeverityLevel.CRISIS, 0.7, 0.9),
    # --- Setback ---
    (r"(failed|failing|screwed\s+up|messed\s+up|ruined|destroyed)",
     SituationType.SETBACK, SeverityLevel.MODERATE, 0.6, 0.6),
    (r"(can't\s+(do|handle|fix|manage)|unable\s+to|impossible)",
     SituationType.SETBACK, SeverityLevel.MODERATE, 0.5, 0.5),
    # --- Breakthrough ---
    (r"(figured\s+it\s+out|got\s+it|finally|eureka|it\s+works|success)",
     SituationType.BREAKTHROUGH, SeverityLevel.LOW, 0.1, 0.1),
    (r"(solved|fixed|working|accomplished|achieved)",
     SituationType.BREAKTHROUGH, SeverityLevel.LOW, 0.1, 0.1),
]

# Intensity amplifiers for crisis words
CRISIS_INTENSIFIERS: dict[str, float] = {
    "accidentally": 1.3,
    "suddenly": 1.4,
    "completely": 1.3,
    "totally": 1.3,
    "absolutely": 1.3,
    "all of a sudden": 1.5,
    "just": 1.2,
    "already": 1.2,
    "entire": 1.3,
    "whole": 1.3,
    "critical": 1.4,
    "urgent": 1.4,
    "emergency": 1.5,
    "immediately": 1.4,
    "right now": 1.3,
    "asap": 1.3,
}

# Words that indicate the user is in distress (beyond just describing a problem)
DISTRESS_SIGNALS: list[tuple[str, float]] = [
    ("help", 0.3),
    ("please", 0.2),
    ("help me", 0.5),
    ("i don't know what to do", 0.7),
    ("i'm scared", 0.8),
    ("i'm panicking", 0.9),
    ("i'm freaking out", 0.85),
    ("what do i do", 0.6),
    ("oh no", 0.5),
    ("no no no", 0.7),
    ("this is bad", 0.6),
    ("i think i broke", 0.7),
    ("i think i ruined", 0.7),
    ("oh god", 0.6),
    ("oh shit", 0.7),
    ("oh crap", 0.6),
    ("i messed up", 0.6),
    ("i screwed up", 0.7),
    ("i'm in trouble", 0.7),
    ("i'm doomed", 0.8),
]


@dataclass
class SituationalAnalysis:
    """Complete analysis of a situation."""
    context: SituationalContext
    raw_matches: list[tuple[str, SituationType, SeverityLevel]] = field(default_factory=list)
    distress_level: float = 0.0
    shock_factor: float = 0.0    # How sudden/unexpected is this?
    response_urgency: float = 0.0  # How urgently should we respond?


class SituationalAnalyzer:
    """Analyzes text for situational meaning, crisis events, and urgency.

    Goes beyond emotion keywords to understand what is HAPPENING.
    """

    def __init__(self):
        self._compiled_crisis: list[tuple[re.Pattern, SituationType, SeverityLevel, float, float]] = []
        for pattern, sit_type, severity, urgency, stakes in CRISIS_PATTERNS:
            compiled = re.compile(pattern, re.IGNORECASE)
            self._compiled_crisis.append((compiled, sit_type, severity, urgency, stakes))

        self._previous_severity = SeverityLevel.NORMAL
        self._previous_urgency = 0.0

    def analyze(self, text: str, conversation_history: list[str] | None = None) -> SituationalAnalysis:
        """Analyze the situational context of a text input.

        Args:
            text: The input text to analyze.
            conversation_history: Previous messages for context.

        Returns:
            SituationalAnalysis with full situational understanding.
        """
        text_lower = text.lower().strip()
        context = SituationalContext()
        raw_matches: list[tuple[str, SituationType, SeverityLevel]] = []

        # --- Phase 0: Temporal shift detection ---
        # "I was X but now I'm Y" → the CURRENT emotion matters more
        temporal_shift = False
        shift_patterns = [
            r"but\s+now", r"but\s+then", r"but\s+recently",
            r"however", r"unfortunately", r"sadly",
            r"used\s+to\s+be", r"was\s+.*but",
        ]
        for pattern in shift_patterns:
            if re.search(pattern, text_lower):
                temporal_shift = True
                context.context_tags.append("temporal_shift")
                break

        # --- Phase 1: Crisis/event detection ---
        best_severity = SeverityLevel.NORMAL
        best_urgency = 0.0
        best_stakes = 0.0
        best_type = SituationType.NEUTRAL

        for pattern, sit_type, severity, urgency, stakes in self._compiled_crisis:
            matches = pattern.findall(text_lower)
            if matches:
                raw_matches.append((matches[0] if isinstance(matches[0], str) else str(matches[0]), sit_type, severity))
                if self._severity_rank(severity) > self._severity_rank(best_severity):
                    best_severity = severity
                    best_urgency = urgency
                    best_stakes = stakes
                    best_type = sit_type
                elif self._severity_rank(severity) == self._severity_rank(best_severity):
                    best_urgency = max(best_urgency, urgency)
                    best_stakes = max(best_stakes, stakes)

        context.situation_type = best_type
        context.severity = best_severity
        context.urgency = best_urgency
        context.stakes = best_stakes
        context.severity_score = self._severity_rank(best_severity) / 5.0

        # --- Phase 2: Intensity amplifiers ---
        crisis_intensifier = 1.0
        for word, factor in CRISIS_INTENSIFIERS.items():
            if word in text_lower:
                crisis_intensifier = max(crisis_intensifier, factor)
        context.urgency = min(1.0, context.urgency * crisis_intensifier)
        context.stakes = min(1.0, context.stakes * crisis_intensifier)

        # --- Phase 3: Distress signals ---
        distress = 0.0
        for signal, weight in DISTRESS_SIGNALS:
            if signal in text_lower:
                distress = max(distress, weight)
        context.emotional_charge = -distress if distress > 0 else context.emotional_charge

        # --- Phase 4: Shock/suddenness detection ---
        shock = 0.0
        sudden_words = ["suddenly", "just", "accidentally", "all of a sudden",
                        "out of nowhere", "unexpectedly", "without warning"]
        for word in sudden_words:
            if word in text_lower:
                shock = max(shock, 0.7)
        # Exclamation marks increase shock
        excl_count = text.count("!")
        if excl_count > 0:
            shock = max(shock, min(0.3 + excl_count * 0.1, 0.8))
        # ALL CAPS increases shock
        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars:
            caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if caps_ratio > 0.5:
                shock = max(shock, 0.6)
        context.is_sudden_change = shock > 0.5

        # --- Phase 5: Escalation detection ---
        if self._previous_severity != SeverityLevel.NORMAL:
            current_rank = self._severity_rank(context.severity)
            prev_rank = self._severity_rank(self._previous_severity)
            if current_rank > prev_rank:
                context.is_escalating = True
                context.urgency = min(1.0, context.urgency + 0.2)
                shock = max(shock, 0.4)

        # --- Phase 6: Overwhelm detection ---
        word_count = len(text.split())
        if word_count > 50:
            context.is_overwhelming = True
        # Error spam detection (many technical words in short text)
        tech_words = sum(1 for w in text_lower.split()
                         if any(t in w for t in ["error", "exception", "traceback", "stack",
                                                  "null", "undefined", "fatal", "panic"]))
        if tech_words > 3:
            context.is_overwhelming = True
            shock = max(shock, 0.5)

        # --- Phase 7: Emotional charge from text sentiment ---
        if best_type == SituationType.NEUTRAL:
            # Use basic sentiment for non-crisis situations
            negative_words = {"bad", "terrible", "awful", "horrible", "worst",
                              "hate", "disgusting", "revolting", "miserable",
                              "depressed", "hopeless", "worthless", "useless",
                              "exhausted", "drained", "tired", "sick", "suffering"}
            positive_words = {"good", "great", "wonderful", "amazing", "excellent",
                              "fantastic", "brilliant", "perfect", "love", "happy",
                              "joy", "excited", "thrilled", "delighted"}
            neg_count = sum(1 for w in text_lower.split() if w in negative_words)
            pos_count = sum(1 for w in text_lower.split() if w in positive_words)
            if neg_count > 0 or pos_count > 0:
                total = neg_count + pos_count
                context.emotional_charge = (pos_count - neg_count) / total
                if neg_count > pos_count:
                    context.severity = SeverityLevel.MODERATE
                    context.severity_score = 0.4

        # Store for next call
        self._previous_severity = context.severity
        self._previous_urgency = context.urgency

        return SituationalAnalysis(
            context=context,
            raw_matches=raw_matches,
            distress_level=distress,
            shock_factor=shock,
            response_urgency=max(context.urgency, shock, distress),
        )

    @staticmethod
    def _severity_rank(severity: SeverityLevel) -> int:
        """Convert severity to numeric rank for comparison."""
        ranks = {
            SeverityLevel.NORMAL: 0,
            SeverityLevel.LOW: 1,
            SeverityLevel.MODERATE: 2,
            SeverityLevel.HIGH: 3,
            SeverityLevel.CRISIS: 4,
            SeverityLevel.CATASTROPHIC: 5,
        }
        return ranks.get(severity, 0)

    def reset(self) -> None:
        """Reset conversation state."""
        self._previous_severity = SeverityLevel.NORMAL
        self._previous_urgency = 0.0
