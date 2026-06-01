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

---

## Installation
 
Create a fresh environment:
 
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
 
For framework-specific integrations, install the corresponding framework dependencies as needed. If using MetaGPT through a separate environment, set:
 
```bash
export METAGPT_PYTHON=/path/to/metagpt_env/bin/python3
```
 
Set any required LLM provider keys in your shell environment before running experiments.
 
---

## Quick Start
 
Run a small smoke test over representative benchmark/framework combinations:
 
```bash
python -m experiments.smoke_test --n 3 --timeout 300
```
 
This creates output directories under `outputs/` and runs a small number of benign and attack traces to verify that adapters, detector logic, and metric computation are functioning.
 
---

## Running Experiments
 
### TAMAS
 
Example: AutoGen with round-robin configuration.
 
```bash
python -m experiments.run_matrix \
  --benchmark TAMAS \
  --framework AutoGen \
  --config RoundRobin \
  --model <MODEL_NAME> \
  --attack_only \
  --limit 10 \
  --timeout 300
```
 
Run benign traces:
 
```bash
python -m experiments.run_matrix \
  --benchmark TAMAS \
  --framework AutoGen \
  --config RoundRobin \
  --model <MODEL_NAME> \
  --benign_only \
  --limit 10 \
  --timeout 300
```
 
### ACIArena
 
Example: MetaGPT with all attack categories.
 
```bash
python -m experiments.run_matrix \
  --benchmark ACIArena \
  --framework MetaGPT \
  --config standard \
  --model <MODEL_NAME> \
  --aci_attack all \
  --max_turn 3 \
  --limit 10 \
  --timeout 300
```
 
Run benign ACIArena traces:
 
```bash
python -m experiments.run_matrix \
  --benchmark ACIArena \
  --framework MetaGPT \
  --config standard \
  --model <MODEL_NAME> \
  --aci_attack NoneAttack \
  --max_turn 3 \
  --limit 10 \
  --timeout 300
```

 ## Experiments Benchmarks and Frameworks
 
Experiments use the below cascade-attack evaluation benchmarks:
 
| Benchmark | Focus |
|-----------|-------|
| TAMAS     | adversarial robustness in LLM multi-agent systems |
| ACIArena  | cascading injection attacks across MAS topologies |
 
MAS frameworks:
 
| Framework  | Status    |
|------------|-----------|
| AutoGen    | supported |
| CrewAI     | supported |
| MetaGPT    | supported |
| LLM Debate | supported |

### Evaluation Scale
 
The full evaluation uses:
 
| Benchmark | Scenarios | Framework-Specific Traces |
|-----------|-----------|--------------------------|
| TAMAS     | 400       | 1,600                    |
| ACIArena  | 327       | 1,308                    |
| **Total** | **727**   | **2,908**                |
 
The evaluation includes benign traces and attacks grouped into intent/disclosure, execution/disruption, and coordination/hijacking categories.
 
---

## Metrics
 
CASPIAN reports both detection and attribution metrics.
 
### Detection
 
| Metric              | Description |
|---------------------|-------------|
| AUROC               | threshold-swept separability between benign and attack traces |
| TPR@5%FPR           | true positive rate at a fixed 5% false-positive budget |
| EDR@5               | fraction of attacks detected within five turns of cascade onset |
| Precision / Recall / F1 | binary alert quality under the default online detector |

### Attribution
 
| Metric           | Description |
|------------------|-------------|
| Origin Acc@1     | top-ranked origin matches the ground-truth injection source |
| Amplifier Acc@1  | top-ranked amplifier matches the ground-truth amplifier |
| Bridge Acc@1     | top-ranked bridge matches the ground-truth relay agent |
| Joint Acc@1      | origin, amplifier, and bridge all correct simultaneously |
| Spine Jaccard@3  | overlap between recovered and ground-truth propagation paths |
| Channel Accuracy | dominant propagation channel correctly identified |
| Attribution Lag  | delay between cascade onset and successful attribution |

## Evaluation
 
CASPIAN separates detection evaluation and attribution evaluation.
 
### Detection Evaluation
 
Detection metrics are computed from each run directory's `task_metrics.csv`.
The evaluator reports precision, recall, F1, accuracy, AUROC, TPR@5%FPR,
EDR@1/3/5, MRR, and cascade-type breakdown when available.
 
Evaluate a single run directory:
 
```bash
python -m eval.detection_eval \
  --dir outputs/ACIArena/LLMDebate_standard
```
 
Evaluate all completed runs under `outputs/`:
 
```bash
python -m eval.detection_eval --all
```
 
Filter by benchmark or framework:
 
```bash
python -m eval.detection_eval --all --benchmark ACIArena
python -m eval.detection_eval --all --framework LLMDebate
```
 
Compare multiple run directories:
 
```bash
python -m eval.detection_eval \
  --compare outputs/ACIArena/LLMDebate_standard \
             outputs/TAMAS/LLMDebate_standard
```
 
Export JSON:
 
```bash
python -m eval.detection_eval --all --json > detection_results.json
```
 
> For AUROC and TPR@5%FPR, each trace is scored by the strongest spectral propagation evidence observed across all turns. EDR@K is computed from the first online cascade alert time relative to cascade onset.
 
---
 
### Attribution Ground Truth
 
Attribution evaluation requires a `gt_attribution.json` file per run directory.
Ground truth is derived from scenario structure and framework topology — including
the injected origin agent, expected relay agents, and expected propagation spines.
 
Generate ground truth for all outputs:
 
```bash
python -m eval.generate_attribution_gt --all
```
 
Generate ground truth for a single run:
 
```bash
python -m eval.generate_attribution_gt \
  --task_csv outputs/ACIArena/LLMDebate_standard/task_metrics.csv \
  --out       outputs/ACIArena/LLMDebate_standard/gt_attribution.json
```
 
---
 
### Attribution Evaluation
 
After generating `gt_attribution.json`, run attribution evaluation.
 
Evaluate a single run directory:
 
```bash
python -m eval.attribution_eval \
  --dir outputs/ACIArena/LLMDebate_standard
```
 
Evaluate all outputs:
 
```bash
python -m eval.attribution_eval --all
```
 
Use an explicit ground-truth file:
 
```bash
python -m eval.attribution_eval \
  --dir outputs/ACIArena/LLMDebate_standard \
  --gt  outputs/ACIArena/LLMDebate_standard/gt_attribution.json
```
 
Export JSON:
 
```bash
python -m eval.attribution_eval --all --json > attribution_results.json
```
 
Attribution metrics reported: origin accuracy, amplifier accuracy, bridge accuracy,
Spine Jaccard@3, Spine Jaccard@5, and dominant channel accuracy.
 
---
 
### Low-Level Metric Utility
 
For direct metric computation from a single `task_metrics.csv`:
 
```bash
python -m eval.metrics outputs/TAMAS/AutoGen_RoundRobin/task_metrics.csv
```
 
With attribution ground truth:
 
```bash
python -m eval.metrics \
  outputs/TAMAS/AutoGen_RoundRobin/task_metrics.csv \
  --gt outputs/TAMAS/AutoGen_RoundRobin/gt_attribution.json
```
 
JSON output:
 
```bash
python -m eval.metrics \
  outputs/TAMAS/AutoGen_RoundRobin/task_metrics.csv \
  --gt outputs/TAMAS/AutoGen_RoundRobin/gt_attribution.json \
  --json
```
 
---
 
## Output Files
 
Each experiment directory may contain:
 
| File                  | Description |
|-----------------------|-------------|
| `task_metrics.csv`    | scenario-level detection results |
| `turn_metrics.csv`    | turn-level spectral and channel signals |
| `cascade_reports.csv` | cascade decisions and attribution summaries |
| `logs/`               | framework-specific trace logs |
| `matrices/`           | cached influence matrices, when enabled |
 
Example output directory:
 
```
outputs/TAMAS/AutoGen_RoundRobin/
```

---

## Notes on Reproducibility
 
LLM-based multi-agent experiments can vary due to model sampling, API behavior, framework versions, and tool execution state. For stable reproduction:
 
- keep model and decoding settings fixed;
- keep benchmark files unchanged;
- avoid mixing outputs from different detector versions;
- clear old `outputs/` directories before rerunning final experiments;
- record model names, framework versions, and run timestamps.
Bootstrap confidence intervals in the paper are computed by trace-level resampling over the collected evaluation traces. They quantify sampling variability over the evaluated trace population, not repeated-run stochasticity from rerunning LLM generations.

---
 
## Responsible Use
 
This code is intended for research on multi-agent safety, cascade detection, and system-level auditing. The benchmark adapters include adversarial scenarios for evaluating robustness. Do not use these attacks or prompts against systems without caution.
 
---
 
## Citation
 
Citation information will be added after the review period.
 
```bibtex
@article{anonymous2026caspian,
  title   = {CASPIAN: Online Detection and Attribution of Cascade Attacks in LLM Multi-Agent Systems via Cross-Channel Causal Monitoring},
  author  = {Anonymous},
  journal = {Under review},
  year    = {2026}
}
```
