# WM HDV Context And Edge-Conditioned Action Plan

## Context

The current new-WM line loads as `hdv_context=False`, even though the MARL rollout path already has code to pass a separate HDV background when the checkpoint supports it. This means the newly trained world model cannot distinguish:

- controllable AV planned edge flow,
- exogenous episode-level HDV response/background,
- departures and time context.

There is also a likely action-conditioning bottleneck in the world model architecture. `ActionDepartGNNEncoder` currently encodes edge-level `action`, `depart`, and optional `hdv_context` with edge GNNs, but then reduces them with global `mean/max` pooling into one ACTION token. That global token is useful as context, but it can erase the spatial pattern that matters for routing: two route-flow fields with similar aggregate load can have very different edge-level congestion consequences.

The first fix should therefore be architectural, not another reward/advantage patch.

## Diagnosis

Current `graph_cond` action path:

1. Encode `av_planned_edge_flow[e]` with an edge GNN.
2. Encode `depart[e]` with another edge GNN.
3. Optionally encode `hdv_context[e]`.
4. Pool each edge tensor with `mean(edge)` and `max(edge)`.
5. Build one global ACTION token from the pooled vectors.
6. Separately inject raw edge action values into spatial tokens when `--edge_action_injection` is enabled.

The raw edge injection helps, but it is not the same as graph-conditioned edge action semantics:

- it only sees raw action values, not the graph-encoded action/depart relationship,
- it does not carry the pooled encoder's topology-aware hidden state per edge,
- HDV response is not enabled in the new full-cycle WM,
- the main semantic action token is still globally pooled.

## Target Design

Keep the existing global ACTION token, but add an edge-aligned conditioning path:

```text
AV planned edge flow[e]
depart[e]
HDV context[e]
static edge features
adjacency
        |
ActionDepartGNNEncoder edge GNNs
        |
per-edge conditioning token C[e]
        |
spatial_token[e] += Linear(C[e])
```

This makes the transition model see the spatial shape of the action field at the same edge token where it predicts next flow.

The global ACTION token remains available for episode-level summary information, but it is no longer the only topology-aware action path.

## Implementation Plan

1. Add `edge_conditioning` support to `ActionDepartGNNEncoder`.
   - Preserve current default behavior for old checkpoints.
   - When enabled, return both the global ACTION token and per-edge conditioning tokens.
   - Build per-edge tokens from graph-encoded action and depart hidden states, plus graph-encoded HDV context when `hdv_context=true`.

2. Inject graph-conditioned edge tokens into `Dynamics` spatial tokens.
   - Add a `graph_cond_edge_conditioning_scale` multiplier.
   - Apply this injection before the transformer, beside the existing raw edge-action injection.
   - Keep existing raw edge action injection enabled because it is a useful direct value channel.

3. Ensure new full-cycle WM training uses HDV context.
   - Set `transition_hdv_context_mode=episode_response`.
   - Set `transition_hdv_context_weight=1.0`.
   - Set `transition_hdv_context_scale=1.0`.
   - Set HDV head semantics to episode response:
     - `hdv_policy_target_mode=episode_mean`
     - `hdv_policy_action_mode=window_mean`

4. Make settings files able to carry these options.
   - Add dynamics-stage defaults for HDV context and graph-conditioned edge conditioning.
   - Generated full-cycle settings should record the WM conditioning choices.

5. Retrain the WM from scratch on the current ratio `0.4` data.
   - Use the conda `URB` Python.
   - Use CUDA outside the sandbox.
   - Confirm the resulting checkpoint prints `hdv_context=True`.

## What This Does Not Do

This plan does not add a new action-sensitivity loss as the primary fix. The first hypothesis to test is that the model's action insensitivity is caused by spatial pooling/bottlenecking, so the fix is to make action conditioning edge-aligned in the transition model.

Action-contrast or additional global reward heads can still be added later if the retrained edge-conditioned WM remains insensitive.

