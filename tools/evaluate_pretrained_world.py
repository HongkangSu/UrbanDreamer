"""Frozen held-out prediction checks after offline pretraining; no policy training."""
from pathlib import Path
from dataclasses import replace
import argparse,json,sys,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from urban_dreamer.workflow.config import load_config
from urban_dreamer.workflow.checkpoint import load_checkpoint,CheckpointPhase
from urban_dreamer.workflow.pipeline import combined_input_hash
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.learning.training import MainlineTrainer
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from tools.migrate_pretrained_meso import install_sparse_codec
from tools.world_model_evaluation import evaluate_world_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--per-method',type=int,default=2)
    args=parser.parse_args()
    torch.set_num_threads(2)
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.08,torch.device(args.device))
    config=load_config(args.run_dir/'run_config.toml')
    prepared=validate_prepared_inputs(config);topology=load_traffic_topology(prepared)
    checkpoint=load_checkpoint(output_dir=args.run_dir,expected_config_fingerprint=config.semantic_fingerprint,expected_input_hash=combined_input_hash(prepared))
    if checkpoint.phase is not CheckpointPhase.ONLINE_TRAINING or checkpoint.next_unit!=0:
        raise ValueError('evaluation requires completed offline pretraining, before online training')
    if checkpoint.training_state['metrics']['world_model_initialization_epochs_completed']!=config.run.world_model_initialization_epochs:
        raise ValueError('incomplete pretraining')
    trainer=MainlineTrainer(config=replace(config,run=replace(config.run,device=args.device)),prepared_inputs=prepared,topology=topology)
    try:
        install_sparse_codec(trainer);trainer.load_state_dict(checkpoint.training_state);trainer.validate_codec_ready()
        dataset=CompactPretrainDataset(config,prepared,topology,4,cache_manifest_path=config.data.validation_cache_manifest or config.data.pretrain_cache_manifest,split="validation")
        rng=np.random.default_rng(9142);methods={};selected=[]
        for i,e in enumerate(dataset.manifest['entries']):methods.setdefault(e['method'],[]).append(i)
        for method,indices in sorted(methods.items()):
            for i in rng.permutation(indices)[:args.per_method]:
                e=dataset.manifest['entries'][int(i)];last=max(0,e['windows']-4)
                starts=sorted(set([min(4,last),min(40,last)]));selected.append(dict(index=int(i),method=method,path=e['path'],demand_group=e['demand_group'],starts=starts))
        def batches():
            for row in selected:yield dataset.batch_from_starts(row['index'],row['starts'],materialization_horizon=4,fm_only=True)
        metrics=evaluate_world_model(trainer,batches(),seed=31415,n_samples=3,horizon=3)
        result=dict(checkpoint=str(checkpoint.path),config_fingerprint=config.semantic_fingerprint,checkpoint_sha256=hashlib.sha256(checkpoint.path.read_bytes()).hexdigest(),validation_manifest_identity=dataset.manifest['identity'],selected=selected,metrics=metrics,notes=['Independent one-window forecasts at three observed offsets, not ninety-second recursive policy rollouts.','Selected demand groups are disjoint from this run training; no labels feed into policy optimization.','Physical decoder speed is compared with its trained endpoint-mean label.'])
        path=args.run_dir/'heldout_prediction_evaluation.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2),flush=True)
    finally:trainer.close()
if __name__=='__main__':main()
