"""Historical one-archive Toy verifier; staged codec validation is now required.

One trajectory cannot provide independent training and held-out demand groups.
This entrypoint fails before collection, model construction or output writes.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True,
                        help='historical one-archive Toy verification input')
    parser.add_argument('--backend', choices=('traci', 'libsumo'), default='traci')
    parser.parse_args(argv)
    parser.error('Historical one-archive Toy verification lacks independent held-out demand '
                 'trajectories and cannot establish raw/cohort codec readiness. Use: '
                 'python tools/train_codec_ready.py --pilot-dir PILOT --output-dir CODEC; '
                 'python tools/build_pretrain_cache.py --config CONFIG --census-dir CENSUS '
                 '--pilot-dir PILOT --output-dir CACHE; python tools/run_complete_pretrain.py '
                 '--config CONFIG --codec-dir CODEC --cache-manifest CACHE/cache_manifest.json '
                 '--output-dir RUN. No collection or training has started.')


if __name__ == '__main__':
    main()
