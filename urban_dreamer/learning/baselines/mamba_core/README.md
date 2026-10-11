# MAMBA network core

These files preserve the torch-only network/configuration blocks from the
repository's sibling `baseline/mamba` adaptation of jbr-ai-labs/mamba (MIT,
Copyright 2022 jbr-ai-labs). This is the AAMAS 2022 multi-agent model-based
algorithm, not the later selective state-space sequence architecture.

The routing adapter uses the discrete categorical RSSM, agent transformer,
recurrent latent actor and attention critic. Communication is bounded to 32
vehicles per chunk, omitting communication across chunks. Realized vehicle
identities are stable within each episode, and weights do not depend on N.

The traffic adaptation replaces per-agent reward averaging with a separate
global RSSM over exact network/vehicle pooled context and one actual team reward
label per 30-second window. Agent dynamics retain all real branch decisions,
including multiple decisions by one vehicle within a window. Imagined active
agents are only the selected origin cohort; no future demand is injected.
The global RSSM statistically models full-network traffic and continues team
credit after individual arrivals, with a real-episode-tail-trained critic.

The global observation contains means of every network flow feature and pooled
physical descriptors over every active vehicle with the original realized-N
denominator. This compression can hide local bottlenecks. The imagined routing
availability/candidate observations are learned; their evolving masks are not
hard graph feasibility guarantees. Real actions always use the shared runner's
actual legal next-hop masks. Imagination stops at the known episode deadline,
with proportionate cost for its final partial window and zero terminal bootstrap.
Frozen imagined behavior is reused for four clipped PPO epochs with the same
episode prior. These routing/global-reward adaptations mean this implementation
preserves the original network family rather than bit-identical upstream code.
