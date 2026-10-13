"""Verify real GPU recovery and CPU export/inference without changing the chosen model."""
import json
from pathlib import Path
import subprocess
import torch
import sys
ROOT=Path(__file__).resolve().parents[1]
GPU=ROOT/'.venv-train-gpu/Scripts/python.exe'
CPU=ROOT/'.venv/Scripts/python.exe'
sys.path.insert(0,str(ROOT))
from edgeprep.challenge import restore_recipe

def run(python,*args):
    subprocess.run([str(python),*args],cwd=ROOT,check=True)

def verify_saved_updates():
    original,_=restore_recipe(ROOT/'artifacts/all_data/resize192/recipe.json')
    state=torch.load(ROOT/'artifacts/gpu_smoke/resize192/recovered_state.pt',map_location='cpu',weights_only=True)
    original=original.state_dict()
    changed=[k for k in state if not torch.equal(state[k],original[k])]
    bn_keys=[k for k in state if k.endswith(('running_mean','running_var','num_batches_tracked'))]
    assert changed, 'GPU recovery did not update any tensors'
    assert all(torch.isfinite(v).all() for v in state.values()), 'Nonfinite saved checkpoint'
    assert all(state[k].device.type=='cpu' for k in state), 'Saved tensors must be CPU portable'
    assert all(torch.equal(state[k],original[k]) for k in bn_keys), 'Frozen BN buffers changed'
    return {'changed_tensor_count':len(changed),'all_checkpoint_tensors_finite':True,
        'bn_buffers_unchanged':True,'checkpoint_tensors_cpu':True}

def main():
    run(GPU,'-m','edgeprep.challenge','recover','--recipe','artifacts/all_data/resize192/recipe.json',
        '--output','artifacts/gpu_smoke/resize192','--train-limit','100','--batch-size','20','--epochs','1','--lr','1e-5')
    run(CPU,'-m','edgeprep.challenge','export','--recipe','artifacts/gpu_smoke/resize192/recipe.json','--output','artifacts/gpu_smoke/export')
    run(CPU,'-m','edgeprep.challenge','evaluate','--model','artifacts/gpu_smoke/export/model_fp32.onnx','--limit','100',
        '--threads','4','--output','results/gpu_training/smoke_accuracy.json')
    recipe=json.loads((ROOT/'artifacts/gpu_smoke/resize192/recipe.json').read_text(encoding='utf-8'))
    recovery=recipe['recovery'];ids=recovery.pop('train_indices')
    result={'status':'completed','purpose':'GPU training / CPU export and inference pipeline verification, not a competition candidate',
        'training_examples':len(ids),'optimizer_steps':5,'training':recovery,
        'checkpoint_sha256':recipe['recovered_sha256'],'cpu_export_recipe':'artifacts/gpu_smoke/export/recipe.json',
        'accuracy_evidence':'results/gpu_training/smoke_accuracy.json','selected_candidate_changed':False,
        'checkpoint_validation':verify_saved_updates()}
    (ROOT/'results/gpu_training/verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
