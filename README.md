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

## Abstract

Cascade attacks in LLM multi-agent systems (MAS) arise when adversarial influence propagates across agents and leads to escalated system-level failures through complex agent interactions. Detecting such cascades is challenging, as their signals are distributed, tightly coupled across interaction channels, and often appear plausibly benign locally but may unfold quickly either within a single turn or gradually across multiple turns. Existing defenses, being largely local and text-centric, fail to capture such cross-channel, temporally coordinated dynamics of cascade propagation. Therefore, we propose **CASPIAN**, the first framework that provides a *unified, cross-channel causal analysis* of cascade behavior in LLM-MAS through online monitoring of dynamic influence propagation across agents. CASPIAN models multiagent interactions using a unified, dynamic causal influence matrix across channels, estimated efficiently via a late-interaction conditional transfer entropy (LI-CTE) formulation, thereby enabling the detection of cascade onset from emergent system-level structure rather than isolated anomalies. It further performs online causal attribution, identifying the origin, bridge, and amplifier agents driving the cascade and reconstructing its principal propagation pathways, capabilities not supported by existing methods. Across diverse multi-agent frameworks and benchmarks, CASPIAN consistently outperforms semantic guardrails, LLM-based judges, and graph-based anomaly detectors in both detection accuracy and early cascade identification while operating with *sub-1% relative overhead latency*. These results demonstrate that unified cross-channel causal modeling is essential for reliably detecting and understanding cascade failures in LLM multi-agent systems.
