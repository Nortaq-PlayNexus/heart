<p align="center">
  <picture>
    <img src="https://img.shields.io/badge/H.E.A.R.T.-EMOTIONAL%20CORE-ff3b3b?style=flat-square&labelColor=0a0e1a" alt="heart" />
  </picture>
</p>

# H.E.A.R.T. :: EMOTIONAL ARCHITECTURE

**Human-like Emotional Architecture — a next-generation AI emotional cognition system.** Perception, appraisal, episodic memory, response generation — in pure Python, no cloud, local-first.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-ffc430?style=flat-square&logo=python&logoColor=ffc430&labelColor=0a0e1a" alt="python"/>
  <img src="https://img.shields.io/badge/LOCAL-ONLY-B8FF1E?style=flat-square&labelColor=0a0e1a" alt="local"/>
  <img src="https://img.shields.io/badge/PURE%20PY-ZERO%20DEPS-3dd5ff?style=flat-square&labelColor=0a0e1a" alt="puredeps"/>
  <img src="https://img.shields.io/badge/VERSION-0.1.0-E8E8E8?style=flat-square&labelColor=0a0e1a" alt="version"/>
  <a href="LICENSE"><img src="https://img.shields.io/badge/LICENSE-MIT-ff3b3b?style=flat-square&labelColor=0a0e1a" alt="license"/></a>
</p>

<pre>
IDENT ......... HEART-01
CLASS ......... AI EMOTIONAL COGNITION CORE
STATUS ........ ONLINE / EXPERIMENTAL
PHYSIOLOGY .... PERCEIVE → APPRAISE → REMEMBER → RESPOND
FREQ .......... EMOTIONAL SPECTRUM (LOCAL)
LINK .......... /heart
</pre>

---

## // 01 :: SIGNAL

H.E.A.R.T. is a machine that **perceives, reasons about, remembers, and responds to emotion** in a human-like way. Not a sentiment label — a cognitive architecture: appraisal theory, episodic memory of encounters, emotional state management, and adaptive response generation.

Everything runs on your hardware. No telemetry, no cloud, no account.

```
        .--.
       |o_o |      HEART.SYS
       |:_/ |      emotional cognition online
      //   \ \
     (|     | )    perceive → appraise → remember → respond
     /'\_   _/`\
    \___)=(___/
```

---

## // 02 :: PHYSIOLOGY (ARCHITECTURE)

```
H.E.A.R.T.
├── core        — engine, configuration, emotional state management
├── perception  — emotion detection, sentiment analysis, feature extraction
├── cognition   — appraisal theory, emotional reasoning, inference
├── memory      — episodic, semantic, and working emotional memory
├── models      — emotion models (Plutchik, dimensional) and personality
├── response    — response generation and emotional strategies
└── utils       — logging, helpers
```

| Module | Purpose |
|--------|---------|
| **Perception** | detect and analyze emotions from text, audio, visual cues |
| **Cognition** | appraise emotional significance, reason about causes and consequences |
| **Memory** | store and retrieve episodic emotional experiences and semantic knowledge |
| **Models** | Plutchik's wheel, dimensional models (valence-arousal), Big Five personality |
| **Response** | generate emotionally appropriate responses with adaptive strategies |

---

## // 03 :: SETUP // WIRE UP THE HEART

```bash
$ pip install -e .
```

---

## // 04 :: TRANSMIT // QUICK START

```python
from heart import HeartEngine

engine = HeartEngine()
result = engine.process("I'm feeling really down today...")

print(result.primary_emotion)
print(result.response)
```

---

## // 05 :: MANIFEST

<details>
  <summary><code>$ cat manifest/structure</code></summary>

See [pyproject.toml](pyproject.toml) for metadata. Dev extras: `pip install -e .[dev]` → `pytest` + `ruff`.

</details>

---

## // 06 :: LEGAL

**License:** [MIT](LICENSE)

---

```
 ┌─────────────────────────────────────────────┐
 │  EMOTIONAL COGNITION // SIGNAL DETECTED     │
 │  HEART-01 // PURE PYTHON, ZERO CLOUD       │
 └─────────────────────────────────────────────┘
END OF TRANSMISSION
```