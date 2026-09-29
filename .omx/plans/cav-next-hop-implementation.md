# Authorized implementation and cleanup plan

Source: user supplied five-part full CAV / event next-hop / fixed 30-second model plan.

1. Lock next-hop candidate/mask, event boundaries, replay T+1/T, elapsed rewards and resumability in focused tests.
2. Replace role/path input artifacts and RouteRL environment with passenger-reachable next hops and direct TraCI subscriptions. Preserve outputs/.
3. Replace policy observations and model/replay conditions together; delete HDV branches and whole-path advancement without compatibility shims.
4. Integrate event imagination, per-vehicle elapsed-time returns, in-transit bootstrap and macro critic updates including empty-decision windows.
5. Run targeted tests, full suite, compile/static checks available without dependencies, and real Toy lifecycle plus short-edge/multicar SUMO checks. Record city microsteps/runtime and remaining approximations.

Boundaries: environment/topology, policy/observations/imagination, world-model/replay, input/config/checkpoint/docs each owned independently; training and integration owned by leader. No new dependencies, no edits to user output directories. Cleanup removes dead HDV/path logic and stale artifacts; retain existing model architecture and phase orchestration.
