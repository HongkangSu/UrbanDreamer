"""TraCI startup fallback for SUMO builds whose dynamic port probe returns None."""

from __future__ import annotations

import os
import socket
from typing import List, Optional


_PATCHED = False


def patch_traci_start_port_fallback() -> bool:
    """Patch ``traci.start`` when SUMO's free-port helper is broken.

    Some SUMO Python builds return ``None`` from ``getFreeSocketPort()``. TraCI
    then forwards ``--remote-port None`` to SUMO, which fails before the episode
    starts. This fallback keeps normal environments untouched and only supplies
    a concrete loopback port when TraCI would otherwise pass ``None``.
    """
    global _PATCHED
    if _PATCHED:
        return True
    try:
        import traci
        from sumolib.miscutils import getFreeSocketPort
    except Exception:
        return False

    try:
        probe = getFreeSocketPort()
    except Exception:
        probe = None
    if probe is not None:
        return False

    try:
        import traci.main as traci_main
    except Exception:
        traci_main = None

    original_start = traci.start

    def _reserve_loopback_port() -> Optional[int]:
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
        except OSError:
            return None
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

    def _iter_fallback_ports() -> List[int]:
        ports: List[int] = []
        seen = set()
        for _ in range(16):
            cand = _reserve_loopback_port()
            if cand is None or cand in seen:
                continue
            seen.add(cand)
            ports.append(cand)
        base_port = 8800 + (os.getpid() % 100)
        for cand in range(base_port, base_port + 32):
            if cand in seen:
                continue
            seen.add(cand)
            ports.append(cand)
        return ports

    def _fallback_start(
        cmd,
        port=None,
        numRetries=60,
        label="default",
        verbose=False,
        traceFile=None,
        traceGetters=True,
        stdout=None,
        doSwitch=True,
    ):
        if port is not None:
            return original_start(
                cmd,
                port=port,
                numRetries=numRetries,
                label=label,
                verbose=verbose,
                traceFile=traceFile,
                traceGetters=traceGetters,
                stdout=stdout,
                doSwitch=doSwitch,
            )
        last_exc = None
        for cand in _iter_fallback_ports():
            try:
                return original_start(
                    cmd,
                    port=cand,
                    numRetries=numRetries,
                    label=label,
                    verbose=verbose,
                    traceFile=traceFile,
                    traceGetters=traceGetters,
                    stdout=stdout,
                    doSwitch=doSwitch,
                )
            except Exception as exc:
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("failed to allocate any fallback TraCI port")

    traci.start = _fallback_start
    if traci_main is not None:
        traci_main.start = _fallback_start
    _PATCHED = True
    print("[INFO] Patched traci.start fallback because getFreeSocketPort() returned None.", flush=True)
    return True
