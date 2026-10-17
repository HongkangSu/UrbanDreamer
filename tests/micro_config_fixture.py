"""Independent micro physics for historical fake-session and internal-lane tests."""
from pathlib import Path
import json
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def write_micro_config(directory: Path,
                       source: Path = ROOT / 'configs/toy_cross_bottleneck.toml') -> Path:
    """Preserve historical micro/next-hop behavior and the network location."""
    text = source.read_text(encoding='utf-8')
    original = tomllib.loads(text)
    network_dir = (source.parent / original['inputs']['network_dir']).resolve()
    text = re.sub(r'^network_dir\s*=.*$',
                  lambda _: f'network_dir = {json.dumps(str(network_dir))}',
                  text, flags=re.MULTILINE)
    text = re.sub(r'^traffic_model\s*=.*$', 'traffic_model = "micro"',
                  text, flags=re.MULTILINE)
    text = re.sub(r'^routing_mode\s*=.*\n?', '', text, flags=re.MULTILINE)
    text = text.replace('[simulation]', '[simulation]\nrouting_mode = "next_hop"')
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{source.stem}_micro.toml'
    path.write_text(text, encoding='utf-8')
    return path
