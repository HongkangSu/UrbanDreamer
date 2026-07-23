# MAMBA baseline core

This package is an adaptation of the MAMBA implementation from
`jbr-ai-labs/mamba` (MIT License, copyright 2022 jbr-ai-labs), accompanying the
paper *Scalable Multi-Agent Model-Based Reinforcement Learning* (AAMAS 2022).
It retains the
discrete RSSM, multi-agent attention transition, learned reward/continuation
models, and imagined PPO actor-critic training, while replacing the original
Ray/Flatland/SMAC runner and dense replay with UrbanDreamer's SUMO-facing API
and sparse, fixed-global-slot replay.

The adaptation targets Python 3.12 and PyTorch 2.5 or newer. It intentionally has no
runtime dependency on Ray, Flatland, SMAC, or Weights & Biases.
