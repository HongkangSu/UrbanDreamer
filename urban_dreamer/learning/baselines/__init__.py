"""Reviewer baseline methods for the toy next-hop routing benchmark.

Each submodule registers its method class in ``METHODS`` at import time.
``build_method`` constructs a method around an ``EpisodeRunner`` context
(see ``common.py``); the runner owns the SUMO session lifecycle while the
method owns the policy/value networks and the per-episode update.
"""
from __future__ import annotations

METHODS: dict[str, type] = {}


def build_method(name, args, context):
    """Construct the registered baseline ``name`` with CLI ``args`` and runner ``context``."""
    if name not in METHODS:
        raise ValueError(f'unknown baseline method {name!r}; available: {sorted(METHODS)}')
    return METHODS[name](args, context)


from . import gnac  # noqa: E402,F401
from . import dreamer  # noqa: E402,F401
from . import mbpo  # noqa: E402,F401
from . import dta  # noqa: E402,F401
from . import ppo  # noqa: E402,F401
from . import iql  # noqa: E402,F401
from . import mamba  # noqa: E402,F401
