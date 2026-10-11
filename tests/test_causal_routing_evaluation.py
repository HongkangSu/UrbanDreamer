"""Evaluation CLI selects the requested city without changing policy defaults."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('causal_routing_evaluation', ROOT/'tools/evaluate_causal_routing.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_city_and_paired_reference_are_explicit():
    args = module.build_parser().parse_args(['--method','ga_routing','--city','coulommiers',
        '--source','/snapshot','--reference','/reference.json'])
    assert args.city == 'coulommiers'
    assert args.source/'configs'/f'{args.city}.toml' == Path('/snapshot/configs/coulommiers.toml')
    assert args.reference == Path('/reference.json')
    assert args.seeds == [10042,10043,10044]


def test_default_retains_ingo_without_implicit_external_reference():
    args = module.build_parser().parse_args(['--method','ga_routing'])
    assert args.city == 'ingolstadt_custom'
    assert args.reference is None
