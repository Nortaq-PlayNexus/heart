# H.E.A.R.T. — Human-like Emotional Architecture

A next-generation AI emotional cognition system enabling machines to perceive, reason about, remember, and respond to emotions in a human-like manner.

## Architecture

```
H.E.A.R.T.
├── core        — Engine, configuration, emotional state management
├── perception  — Emotion detection, sentiment analysis, feature extraction
├── cognition   — Appraisal theory, emotional reasoning, inference
├── memory      — Episodic, semantic, and working emotional memory
├── models      — Emotion models (Plutchik, dimensional) and personality
├── response    — Response generation and emotional strategies
└── utils       — Logging, helpers
```

## Core Components

| Module | Purpose |
|--------|---------|
| **Perception** | Detect and analyze emotions from text, audio, visual cues |
| **Cognition** | Appraise emotional significance, reason about causes and consequences |
| **Memory** | Store and retrieve episodic emotional experiences and semantic knowledge |
| **Models** | Plutchik's wheel, dimensional models (valence-arousal), Big Five personality |
| **Response** | Generate emotionally appropriate responses with adaptive strategies |

## Quick Start

```python
from heart import HeartEngine

engine = HeartEngine()
result = engine.process("I'm feeling really down today...")
print(result.primary_emotion)
print(result.response)
```

## Install

```bash
pip install -e .
```

## License

MIT
