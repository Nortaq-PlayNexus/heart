"""Tests for heart.perception modules."""

from heart.perception.detector import EmotionDetector, DetectionResult
from heart.perception.analyzer import SentimentAnalyzer, SentimentAnalysis
from heart.perception.features import FeatureExtractor, TextFeatures
from heart.models.emotion_model import EmotionScores, PlutchikEmotion


class TestEmotionDetector:
    def test_detect_positive(self):
        detector = EmotionDetector()
        result = detector.detect("I am so happy and joyful today!")
        assert isinstance(result, DetectionResult)
        assert result.confidence >= 0
        assert result.word_count > 0

    def test_detect_negative(self):
        detector = EmotionDetector()
        result = detector.detect("I feel very sad and depressed.")
        assert isinstance(result, DetectionResult)
        assert result.confidence >= 0

    def test_detect_neutral(self):
        detector = EmotionDetector()
        result = detector.detect("The weather is cloudy today.")
        assert isinstance(result, DetectionResult)

    def test_detect_empty(self):
        detector = EmotionDetector()
        result = detector.detect("")
        assert isinstance(result, DetectionResult)
        assert result.word_count == 0

    def test_emotion_scores(self):
        detector = EmotionDetector()
        result = detector.detect("I am absolutely thrilled and excited!")
        assert isinstance(result.emotions, EmotionScores)
        assert result.emotions.valence != 0 or result.emotions.arousal != 0

    def test_negation_detection(self):
        detector = EmotionDetector()
        result = detector.detect("I am not happy at all.")
        assert result.negation_detected is True


class TestSentimentAnalyzer:
    def test_positive_sentiment(self):
        analyzer = SentimentAnalyzer()
        scores = EmotionScores()
        scores.valence = 0.8
        scores.arousal = 0.6
        result = analyzer.analyze(scores, "I love this!")
        assert isinstance(result, SentimentAnalysis)
        assert result.label == "positive"
        assert result.score == 0.8

    def test_negative_sentiment(self):
        analyzer = SentimentAnalyzer()
        scores = EmotionScores()
        scores.valence = -0.7
        scores.arousal = 0.4
        result = analyzer.analyze(scores, "I hate this.")
        assert result.label == "negative"

    def test_neutral_sentiment(self):
        analyzer = SentimentAnalyzer()
        scores = EmotionScores()
        scores.valence = 0.0
        scores.arousal = 0.1
        result = analyzer.analyze(scores, "It is what it is.")
        assert result.label == "neutral"

    def test_to_dict(self):
        analyzer = SentimentAnalyzer()
        scores = EmotionScores()
        scores.valence = 0.5
        result = analyzer.analyze(scores, "test")
        d = result.to_dict()
        assert "label" in d
        assert "score" in d


class TestFeatureExtractor:
    def test_extract_basic(self):
        extractor = FeatureExtractor()
        features = extractor.extract("Hello world!")
        assert isinstance(features, TextFeatures)
        assert features.word_count == 2
        assert features.sentence_count == 1

    def test_extract_exclamation(self):
        extractor = FeatureExtractor()
        features = extractor.extract("Amazing!!!")
        assert features.exclamation_count == 3
        assert features.exclamation_ratio > 0

    def test_extract_question(self):
        extractor = FeatureExtractor()
        features = extractor.extract("How are you?")
        assert features.question_count == 1

    def test_extract_uppercase(self):
        extractor = FeatureExtractor()
        features = extractor.extract("THIS IS IMPORTANT")
        assert features.uppercase_ratio > 0.5

    def test_extract_empty(self):
        extractor = FeatureExtractor()
        features = extractor.extract("")
        assert features.word_count == 0

    def test_to_dict(self):
        extractor = FeatureExtractor()
        features = extractor.extract("Hello!")
        d = features.to_dict()
        assert "word_count" in d
        assert "has_emphasis" in d
