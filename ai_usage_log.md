# AI Usage Log

Per Q7.4: prompts used, and which code is AI-generated vs. human-written, for CS4.406
Assignment 1. Updated continuously as work progresses; entries are in chronological order.

## Tooling

Claude Code (Anthropic), used interactively in the terminal against this repo.

## Log

### 2026-08-23 — Part 0: repo scaffolding & setup plan

- Prompt (paraphrased): asked Claude to read `A1.pdf` in full, cross-reference the two
  pre-existing analysis notebooks (`ebnerd_analysis.ipynb`, `mind_analysis.ipynb`), and produce
  a plan for Part 0 (datasets & setup), noting the code submission target is a plain git repo
  (not GitHub Classroom) and asking for a compute recommendation (local GPU vs. Kaggle).
- AI-generated: this file, `.gitignore`, `requirements.txt`, `requirements-gpu.txt`,
  `scripts/download_ebnerd.py`, `scripts/download_mind.py`, README scaffold, directory layout.
- Human-written: none yet at this stage (setup only).
- Human review: reviewed and approved the plan before implementation (see
  `/home/aayush/.claude/plans/fluttering-snacking-pumpkin.md` for the plan as approved).
