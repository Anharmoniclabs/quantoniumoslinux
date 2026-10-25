# Origin Record — Resonant Fourier Transform / quantoniumos

> This document is a permanent record for the founder.
> It is not a pitch deck, not a grant proposal, not a paper.
> It is the honest story of how this work came to exist.

---

## Who Built This

**Luis M. Minier** — New York City.

GED. A handful of college credits. A few Coursera cybersecurity certs, mostly completed with AI help.
No degree in mathematics, physics, computer science, or engineering.
No university lab. No corporate backing. No team.

Primary background: **self-taught musician, visual artist, photographer.**
The same trial-and-error, no-school, iterate-until-it-works ethic that built those skills is
exactly what built this.

---

## How It Started

**Late 2024.** Working a public safety job. Started learning to code through AI — not from
a curriculum, but from building a clothing brand app for himself. Learned the hard way
what tokens are, what you should never expose, how APIs work. Never touched code before that.

That felt rewarding in a specific way: **failing constantly and ending up with something that ran.**

Started building a toy qubit visualizer as a prompt engineering resume project. Hit a wall —
didn't understand quantum, didn't understand the math, didn't know the order of operations.
But that lack of knowledge turned out to matter. It meant asking questions that people with
training don't ask.

**The actual moment:** Making a beat in FL Studio between coding sessions. Pressing a key on a
plastic MIDI keyboard and hearing a continuous tone come out through a wire.

The question that changed everything:

> *"If this plastic keyboard can produce continuous, tangible tone data through a wire —
> why can't we compute in that same continuous space?"*

That led to the ASCII bottleneck. Characters and numbers still had high byte counts.
Then: what can a `.` or a `)` hold? Can punctuation be both 0 and 1 — a wave container
with actual data inside it?

The ideas grew from there. That was the origin.

---

## The Build Timeline

| Date | Event |
|---|---|
| **Jan 11, 2025** | Symbolic Resonance Computing born. First notes, first principles. |
| **Jan 26, 2025** | USPTO Provisional filed — #63/749,644 |
| **Feb 4, 2025** | First public LinkedIn post explaining the framework |
| **Mar 23, 2025** | Zenodo preprint published — DOI: 10.5281/zenodo.15163648 |
| **Apr 2, 2025** | IEEE rejection — no technical refutation |
| **Apr 6, 2025** | USPTO Non-Provisional filed — #19/169,399, 4 claims + math amendment |
| **Oct 2025** | C++ backend, RFTMW-Hybrid O(N log N), FPGA synthesis attempt |
| **Mar 2026** | 101 theorem tests passing. CWRU bearing fault benchmark. KV-cache benchmark. Theorem E formally proven via RIP. |

From first idea to legally protected and publicly timestamped framework:
**less than 4 months.**

---

## What Was Used to Build It

- **AI coding assistants** — GPT-4, Gemini, Copilot — as compilers for natural language intent
- **Iterative development** — idea → full code → test → error → revise → repeat, thousands of times
- **No prior math knowledge** — algebra barely understood; calculus not studied
- **No physics background** — quantum mechanics unknown at start
- **Music intuition** — the concept of continuous tone, phase, resonance, frequency as data
  came from making music, not from textbooks
- **Artist's work ethic** — the same process used to learn music and photography by doing:
  fail, observe, adjust, try again

The AI did not have the idea. The AI compiled the idea into runnable code.
**The architecture, the direction, the questions — those came from the human.**

---

## What Was Built

A novel unitary transform — the **Resonant Fourier Transform (RFT)** — that uses
golden-ratio (φ) frequency spacing instead of integer harmonics.

Formally proven properties (as of March 2026):
- Exact unitarity at finite N (Theorem 2)
- Not a DFT, NUDFT, LCT, or FrFT variant (Theorem 6)
- Spectral concentration gap vs FFT: O(log N) vs N^0.75 (Theorem 8)
- Empirical RIP constant δ₈ = 0.1603 < 0.2929 Candès-Tao threshold (Theorem E)
- 101 deterministic theorem tests passing in CI

Industrial application benchmarked (CWRU SKF6205 bearing, March 2026):
- RFT: 45–52 dB SNR, 0% false alarm, stable under ±5% shaft speed drift
- STFT (industry standard): 0 dB SNR, 33% false alarm rate

AI/LLM application benchmarked (KV-cache positional encoding, March 2026):
- RFT vs RoPE: 47–80% cross-window leakage reduction across 11 configurations

Filed under USPTO Application **19/169,399**.

---

## Why This Document Exists

Because this work was built inside someone else's infrastructure.

The math runs on their machines.
The code was compiled by their models.
The papers go through their review processes.
The distribution lives on their platforms.

None of that changes who had the idea or why it was had.

This document is the record that when people ask —
*"where did the Resonant Fourier Transform come from?"* —
the answer is:

A guy in New York who was making a beat one afternoon and asked a question
nobody else thought to ask, because he didn't know enough to know he wasn't supposed to.

---
March 2026*
*Author: Luis M. Minier*
*USPTO Application 19/169,399*
*Zenodo: 10.5281/zenodo.15163648*
