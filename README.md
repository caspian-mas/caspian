# CASPIAN: Online Detection and Attribution of Cascade Attacks in LLM Multi-Agent Systems via Cross-Channel Causal Monitoring
 
> This repository accompanies an anonymous NeurIPS submission. Author-identifying information has been removed for double-blind review.
 
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Review%20Artifact-6B7280" />
  <img src="https://img.shields.io/badge/LLM--MAS-Safety-111827" />
  <img src="https://img.shields.io/badge/Cascade-Detection-B31B1B" />
</p>

CASPIAN is an online framework for detecting and attributing **cascade attacks** in LLM-based multi-agent systems. Instead of inspecting messages in isolation, CASPIAN tracks how influence propagates across agents, turns, and interaction channels, including *communication, memory, tool use, and execution metadata*.

<p align="center">
  <img src="images/caspian-method.png" width="1200">
</p>

## Overview
 
LLM multi-agent systems can fail through distributed cascade behavior: an adversarial instruction, corrupted memory artifact, unsafe tool result, or misleading intermediate response may propagate across agents and become amplified over time. These failures are difficult to detect with local message-level guardrails because individual messages can appear benign while the system-level interaction structure is becoming unstable.
 
CASPIAN addresses this by modeling cascade attacks as **structural propagation events**. At every turn, it:
 
1. extracts communication, memory, tool, and execution events from MAS logs;
2. estimates directed source-to-target influence using late-interaction conditional transfer entropy;
3. constructs a unified cross-channel influence tensor;
4. monitors spectral amplification, synchronization, phase shifts, and cross-channel spread;
5. detects abrupt and gradual cascade onset online;
6. attributes the cascade to responsible agents and propagation spines.

## Method Summary
 
CASPIAN has three main components.
 
### 1. Cross-Channel Influence Estimation
 
Native MAS traces are converted into directed channel events:
 
```text
(source agent, target agent, channel, payload, features)
```
 
The supported channels are:
 
| Channel | Captures |
|---------|----------|
| `comm`  | agent messages, debate responses, role outputs |
| `mem`   | memory reads/writes, persistent artifact reuse |
| `tool`  | tool/API calls, arguments, returned outputs |
| `exec`  | token usage, latency, completion status, errors |
 
For every feasible agent pair and channel, CASPIAN estimates whether source-agent activity explains downstream target-agent behavior beyond the target's previous channel history.
 
### 2. Online Spectral Cascade Detection
 
CASPIAN monitors the evolving influence topology using spectral propagation signals:
 
| Signal | Meaning |
|--------|---------|
| amplification     | growth in dominant propagation energy |
| gap contraction   | increasing coupling between propagation modes |
| phase shift       | abrupt structural transition in influence dynamics |
| cross-channel spread | propagation distributed across multiple modalities |
| weak-link feasibility | existence of a viable end-to-end propagation route |
 
These signals are combined into an online detector for both single-turn and multi-turn cascades.
 
### 3. Online Attribution
 
Upon detection, CASPIAN uses cached influence matrices to recover:
 
| Role      | Interpretation |
|-----------|----------------|
| Origin    | agent where abnormal influence first enters |
| Bridge    | agent that redistributes influence across pathways |
| Amplifier | agent that reinforces propagation most strongly |
| Spine     | dominant source-to-target propagation path |
| Channel   | primary interaction modality of each spine |
 
Attribution is performed online without replaying the full trace or invoking additional LLM calls.

---
 
## Repository Structure
 
```
.
├── core/                     # CASPIAN detector, spectral logic, influence utilities
├── adapters/                 # Benchmark and framework adapters
├── experiments/              # Experiment runners and smoke tests
├── eval/                     # Detection and attribution metrics
├── aciarena/                 # Cloned ACIArena benchmark
├── TAMAS/                    # Cloned TAMAS benchmark
├── outputs/                  # Generated outputs from local runs
├── crewai_trace_runner.py    # CrewAI execution logger
├── metagpt_trace_runner.py   # MetaGPT execution logger
├── requirements.txt          # Dependencies to install
├── README.md                 

```
 
> The repository is being actively cleaned and updated for the review artifact. Some paths may change slightly as the codebase is finalized.
 
---
