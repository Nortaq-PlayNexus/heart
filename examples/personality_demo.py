"""Personality profile demo for H.E.A.R.T."""

from heart import HeartEngine
from heart.models.personality import PersonalityProfile, TraitVector, BigFiveTrait


def make_profile(
    name: str,
    openness: float = 0.5,
    conscientiousness: float = 0.5,
    extraversion: float = 0.5,
    agreeableness: float = 0.5,
    neuroticism: float = 0.5,
) -> PersonalityProfile:
    profile = PersonalityProfile(
        traits=TraitVector(
            openness=openness,
            conscientiousness=conscientiousness,
            extraversion=extraversion,
            agreeableness=agreeableness,
            neuroticism=neuroticism,
        )
    )
    return profile


def test_with_profile(name: str, profile: PersonalityProfile, text: str):
    engine = HeartEngine()
    engine.personality = profile
    result = engine.respond(text)
    print(f"\n[{name}] Input: {text}")
    print(f"  Response: {result['response_text']}")
    print(f"  Emotion: {result['primary_emotion']} ({result['intensity']:.2f})")


def main():
    print("=" * 60)
    print("H.E.A.R.T. - Personality Demo")
    print("=" * 60)

    introvert = make_profile(
        "Introvert",
        extraversion=0.2,
        neuroticism=0.6,
        agreeableness=0.7,
    )

    extrovert = make_profile(
        "Extrovert",
        extraversion=0.9,
        neuroticism=0.3,
        agreeableness=0.5,
    )

    anxious = make_profile(
        "Anxious",
        neuroticism=0.9,
        extraversion=0.3,
        openness=0.4,
    )

    text = "I just got some really unexpected news that changed everything."

    test_with_profile("Introvert", introvert, text)
    test_with_profile("Extrovert", extrovert, text)
    test_with_profile("Anxious", anxious, text)

    print("\n" + "=" * 60)
    print("Comparing responses to negative input:")
    print("=" * 60)

    negative_text = "I failed the exam and I'm devastated."

    test_with_profile("Introvert", introvert, negative_text)
    test_with_profile("Extrovert", extrovert, negative_text)
    test_with_profile("Anxious", anxious, negative_text)


if __name__ == "__main__":
    main()
