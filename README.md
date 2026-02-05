# ConstraintSpace

ConstraintSpace is a **human-governed decision engine** for creative AI systems.
It evaluates, scores, and selects creative ideas under explicit, creator-specific constraints.

The system is designed so that **machine learning augments decision-making**, rather than silently replacing human judgment.

---

## What Problem This Solves

Most creative AI systems optimize generation first.

ConstraintSpace answers a different question:

> *Given a creator’s identity and constraints, which ideas should be pursued, experimented with, or rejected — and why?*

---

## Core Principles

- Human review is explicit and blocking
- Constraints are first-class artifacts
- Decisions are explainable
- Governance precedes automation
- ML is a replaceable scoring component

---

## High-Level Architecture

```
Profiling → ConstraintSpace → Selection Plan
```

---

## Profiling

**Purpose**

- Build a behavioral profile of a creator from real data

**Properties**

- Batch
- Deterministic
- No human interaction

**Planned Signals (v1+)**
- Computer vision–based format detection
- OCR density (text-heavy vs visual)
- Face presence / shot structure
- Audio energy and speech patterns


**Output**

```
profiling/reviewed/{creator}_profile.yaml
```

ConstraintSpace does not proceed without approval.

---

## Human Review (First-Class Control)

**Purpose**
- Preserve human authority over creator identity
- Prevent silent model drift
- Validate inferred patterns before automation

**Properties**
- Blocking
- Explicit approval or rejection
- Manual edits outside code
- Required for all downstream decisions

**Output**
```
profiling/reviewed/{creator}_profile.yaml
```

## ConstraintSpace

**Responsibilities**

- Compile creator constraints
- Evaluate candidate ideas
- Score and rank ideas
- Apply experiment policy
- Produce an actionable plan

**Entry Point**

```bash
python -m engine.run_constraint_space
```

## Idea Evaluation 

Ideas are evaluated as **proposals**, not generated content.

Example idea schema:

```yaml 
idea_id: idea_001
title: "Explaining X in under 30 seconds"
format_hypothesis:
  duration_sec: 25
  has_voice: true
  has_text: true
  format: talking_head
```

## Evaluation Output 

Each idea produces:

```yaml 
decision: pass | warn | reject
score: 0.0 – 1.0
confidence_band: low | medium | high
reasons: [...]
signals: {...}
```

## Selection Plan (v0) 

ConstraintSpace outputs a concrete plan:

```text 
APPROVED:
- Explaining X in under 30 seconds (0.80)

EXPERIMENTS:
- Explaining X in 10 minutes (0.76)

REJECTED:
```
This completes the decision loop.

## Project Structure

```text
ConstraintSpace/
├── engine/
│   ├── run_constraint_space.py
│   └── human_review.py
├── profiling/
│   ├── run_pipeline.py
│   ├── drafts/
│   ├── reviewed/
│   └── ingestion/
├── constraint_space/
│   ├── build_constraints.py
│   ├── evaluate_idea.py
│   ├── select_ideas.py
│   └── ideas/
```
## Status
- v0 complete 
- Deterministic 
- Explainable 
- ML- Ready 

## Roadmap

### v1 - ML- Assisted Scoring 
-	Replace heuristic scoring with learned models
-	Add CV-based format classification in profiling
-	Train on (creator, idea, outcome) data
-	Preserve human review and decision policies

### v2 - Adaptive Constraints 
- Update constraints from performance feedback
-	Dynamically budget experimentation
-	Calibrate risk tolerance per creator

## Summary 
ConstraintSpace is a decision system, not a generator.
It formalizes creative judgment before automation.

---
