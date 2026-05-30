# Current-to-Destination Route Choice Visualization

All three runs use free-flow generated current-to-destination candidate paths.
The maps color selected candidate path edge usage, not all live vehicle trajectories.

## Runs

- `policy_ep400`: t_all=4.3450, rows=2800, routes={'route0': 2484, 'route1': 249, 'route2': 46, 'route3': 21}
- `freeflow_greedy`: t_all=4.4082, rows=2974, routes={'route0': 2970, 'route1': 0, 'route2': 0, 'route3': 4}
- `live_current_greedy`: t_all=4.3767, rows=2882, routes={'route0': 2533, 'route1': 260, 'route2': 66, 'route3': 23}

## Files

- `*_selected_edge_usage.png`: selected path usage heatmap per run.
- `policy_ep400_minus_freeflow_greedy.png`: red means more policy usage; blue means more free-flow greedy usage.
- `policy_ep400_minus_live_current_greedy.png`: red means more policy usage; blue means more live-current greedy usage.
- `live_current_minus_freeflow_greedy.png`: red means more live-current greedy usage; blue means more free-flow greedy usage.
- `selected_route_counts_by_tick.png`: stacked selected route slot counts over decision ticks.
- `nonzero_route_examples.csv`: sample non-route0 decisions with selected path edges.
- `summary.json`: machine-readable metrics and top selected edges.
