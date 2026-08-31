"""Basic usage example for H.E.A.R.T."""

from heart import HeartEngine


def main():
    engine = HeartEngine()

    test_inputs = [
        "I'm feeling really happy today! Everything is going well.",
        "I can't believe this happened. I'm so frustrated and angry.",
        "What if everything goes wrong? I'm worried about the outcome.",
        "That was completely unexpected! I didn't see it coming at all.",
        "I feel so alone and sad right now.",
        "I trust that things will work out eventually.",
        "I'm really looking forward to the weekend plans!",
        "The meeting was absolutely terrible. I hated every minute.",
    ]

    print("=" * 60)
    print("H.E.A.R.T. - Basic Usage Example")
    print("=" * 60)

    for text in test_inputs:
        result = engine.respond(text)
        state = result["emotional_state"]

        print(f"\nInput:     {text}")
        print(f"Emotion:   {result['primary_emotion']} (intensity: {result['intensity']:.2f})")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Valence:   {state['valence']:.3f}")
        print(f"Response:  {result['response_text']}")
        print("-" * 60)

    print("\n--- Emotional Profile ---")
    profile = engine.get_emotional_profile()
    for key, value in profile.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    print("\n--- Memory Status ---")
    print(f"  Working memory:  {engine.working_memory.size}/{engine.working_memory.capacity}")
    print(f"  Episodic memory: {engine.episodic_memory.size}/{engine.episodic_memory.capacity}")
    print(f"  Semantic memory: {engine.semantic_memory.size} contexts")


if __name__ == "__main__":
    main()
