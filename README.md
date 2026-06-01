# CASPIAN
 
**Online Detection and Attribution of Cascade Attacks in LLM Multi-Agent Systems via Cross-Channel Causal Monitoring**
 
> This repository accompanies an anonymous NeurIPS submission. Author-identifying information has been removed for double-blind review.
 
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Review%20Artifact-6B7280" />
  <img src="https://img.shields.io/badge/LLM--MAS-Safety-111827" />
  <img src="https://img.shields.io/badge/Cascade-Detection-B31B1B" />
</p>

CASPIAN is an online framework for detecting and attributing **cascade attacks** in LLM-based multi-agent systems. Instead of inspecting messages in isolation, CASPIAN tracks how influence propagates across agents, turns, and interaction channels, including *communication, memory, tool use, and execution metadata*.
 
The framework constructs a dynamic cross-channel influence topology, monitors spectral propagation signals online, and attributes detected cascades to origin, bridge, amplifier, and dominant propagation paths.

<p align="center">
  <img src="images/caspian-method.png" width="1200">
</p>
