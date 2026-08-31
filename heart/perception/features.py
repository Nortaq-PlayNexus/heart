from dataclasses import dataclass, field
from typing import Optional
import re


@dataclass
class TextFeatures:
    word_count: int = 0
    sentence_count: int = 0
    exclamation_count: int = 0
    question_count: int = 0
    exclamation_ratio: float = 0.0
    question_ratio: float = 0.0
    uppercase_ratio: float = 0.0
    punctuation_density: float = 0.0
    emoji_count: int = 0
    punctuation_marks: int = 0
    avg_word_length: float = 0.0
    sentence_length_avg: float = 0.0
    has_negation: bool = False
    has_emphasis: bool = False
    sentiment_hint: float = 0.0
    emotional_keywords_count: int = 0

    def to_dict(self) -> dict:
        return {
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "exclamation_count": self.exclamation_count,
            "question_count": self.question_count,
            "exclamation_ratio": self.exclamation_ratio,
            "question_ratio": self.question_ratio,
            "uppercase_ratio": self.uppercase_ratio,
            "punctuation_density": self.punctuation_density,
            "emoji_count": self.emoji_count,
            "punctuation_marks": self.punctuation_marks,
            "avg_word_length": self.avg_word_length,
            "sentence_length_avg": self.sentence_length_avg,
            "has_negation": self.has_negation,
            "has_emphasis": self.has_emphasis,
            "sentiment_hint": self.sentiment_hint,
            "emotional_keywords_count": self.emotional_keywords_count,
        }


class FeatureExtractor:
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )

    def extract(self, text: str) -> TextFeatures:
        features = TextFeatures()
        features.word_count = len(text.split())

        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        features.sentence_count = max(len(sentences), 1)

        features.exclamation_count = text.count("!")
        features.question_count = text.count("?")

        if features.word_count > 0:
            features.exclamation_ratio = features.exclamation_count / features.word_count
            features.question_ratio = features.question_count / features.word_count
            features.avg_word_length = sum(len(w) for w in text.split()) / features.word_count
            features.sentence_length_avg = features.word_count / features.sentence_count

        total_chars = max(len(text), 1)
        features.uppercase_ratio = sum(
            1 for c in text if c.isupper() and c.isalpha()
        ) / total_chars

        features.punctuation_marks = sum(
            1 for c in text if c in "!?.,;:()-'\""
        )
        features.punctuation_density = features.punctuation_marks / total_chars

        emojis = self.EMOJI_PATTERN.findall(text)
        features.emoji_count = len(emojis)

        features.has_negation = any(
            nw in text.lower().split()
            for nw in {
                "not", "no", "never", "neither", "nobody", "nothing",
                "can't", "won't", "don't", "doesn't", "isn't", "aren't",
            }
        )

        features.has_emphasis = features.exclamation_count > 1 or features.uppercase_ratio > 0.1

        emotional_words = 0
        sentiment_sum = 0.0
        for word in text.lower().split():
            clean = word.strip(".,!?;:()\"'")
            if any(
                clean in kw_list
                for kw_list in [
                    "happy glad pleased delighted ecstatic thrilled elated jubilant cheerful excited wonderful amazing love grateful thankful blessed euphoric overjoyed content satisfied blissful radiant smile laugh fun enjoy".split(),
                    "sad sorrow grief depressed miserable heartbroken lonely despair hopeless gloomy melancholy crying tearful devastated mourn loss pain hurt sorry upset down low empty numb disappointed".split(),
                    "angry furious rage outraged irritated annoyed mad livid enraged frustrated hostile resentful bitter aggressive violent indignant infuriated provoked offended displeased resent".split(),
                    "afraid scared anxious worried terrified panic dread nervous fear horror threat danger unsafe insecure shaky tremble panicicked frightened uneasy apprehensive stress nervous".split(),
                ]
            ):
                emotional_words += 1
                if clean in "happy glad pleased delighted ecstatic thrilled elated jubilant cheerful excited wonderful amazing love grateful thankful blessed euphoric overjoyed content satisfied blissful radiant smile laugh fun enjoy".split():
                    sentiment_sum += 1.0
                elif clean in "sad sorrow grief depressed miserable heartbroken lonely despair hopeless gloomy melancholy crying tearful devastated mourn loss pain hurt sorry upset down low empty numb disappointed".split():
                    sentiment_sum -= 1.0

        features.emotional_keywords_count = emotional_words
        features.sentiment_hint = sentiment_sum / max(features.word_count, 1) if features.word_count > 0 else 0.0

        return features
