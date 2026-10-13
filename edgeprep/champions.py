"""Preserve selected experiment winners as versioned, hash-verified model bundles."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path, PureWindowsPath
import re
import shutil
from .challenge_data import digest

ROOT=Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(name, accuracy_path, benchmark_path, reason, destination=None):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*',name):
        raise ValueError('Use a simple lowercase experiment name')
    if not reason.strip():raise ValueError('Document why this is the selected winner')
    accuracy=read(accuracy_path);benchmark=read(benchmark_path)
    if accuracy['examples'] != 27000 or accuracy.get('evaluation_scope') != 'training_set':
        raise ValueError('A challenge winner requires full 27000-image evaluation; diagnostic subsets are not eligible')
    model=Path(accuracy['model']);model_hash=digest(model)
    if accuracy['model_sha256'] != model_hash or benchmark.get('sha256') != model_hash:
        raise ValueError('Accuracy, runtime and model hashes must match')
    if benchmark.get('provider') != 'CPUExecutionProvider':
        raise ValueError('The selected laptop model requires CPU runtime evidence')
    recipe=read(model.parent/'recipe.json')
    checkpoint=None
    if recipe.get('recovered_checkpoint'):
        checkpoint=Path(recipe['recovered_checkpoint'])
        if not checkpoint.exists():checkpoint=model.parent/'recovered_state.pt'
        if digest(checkpoint) != recipe['recovered_sha256']:
            raise ValueError('Recovered checkpoint differs from evaluated recipe')
    base=Path(destination) if destination else ROOT/'best_models'
    bundle=base/f'{name}_{model_hash[:12]}'
    expected={'model_weight16.onnx' if recipe.get('weight_storage') else 'model.onnx':model}
    if checkpoint:expected['recovered_state.pt']=checkpoint
    # Old winner snapshots are never overwritten by a different set of weights.
    for filename,source in expected.items():
        target=bundle/filename
        if target.exists() and digest(target)!=digest(source):
            raise ValueError(f'Snapshot already contains different weights: {target}')
    bundle.mkdir(parents=True,exist_ok=True)
    files={}
    for filename,source in expected.items():
        target=bundle/filename
        if not target.exists():shutil.copy2(source,target)
        assert digest(target)==digest(source)
        files[filename]={'sha256':digest(target),'bytes':target.stat().st_size}
    portable=copy.deepcopy(recipe)
    portable['source_checkpoint']='Models/'+PureWindowsPath(recipe['source_checkpoint']).name
    if checkpoint:portable['recovered_checkpoint']='recovered_state.pt'
    if portable.get('weight_storage'):
        portable['weight_storage'].pop('source_onnx',None)
        portable['weight_storage']['source_onnx_note']='Intermediate FP32 export can be recreated from the saved checkpoint and recipe.'
    (bundle/'recipe.json').write_text(json.dumps(portable,indent=2),encoding='utf-8')
    metrics={'dataset_accuracy':{k:v for k,v in accuracy.items() if k not in ['evaluation_indices','model']},
        'cpu_benchmark':{k:v for k,v in benchmark.items() if k not in ['model','accuracy','per_class_accuracy','validation_examples']},
        'accuracy_evidence':str(Path(accuracy_path).resolve().relative_to(ROOT)),
        'benchmark_evidence':str(Path(benchmark_path).resolve().relative_to(ROOT))}
    (bundle/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    manifest={'name':name,'saved_utc':datetime.now(timezone.utc).isoformat(),'selection_reason':reason,
        'selection_method':'Explicit experiment winner; balance accuracy, worst-class recall, size, CPU latency and estimated energy. Not highest accuracy alone.',
        'files':files,'inference_model':next(iter(expected)),
        'training_checkpoint':'recovered_state.pt' if checkpoint else None,
        'recipe':'recipe.json','metrics':'metrics.json',
        'weight_precision':recipe.get('weight_storage',{}).get('weight_storage_precision','FP32'),
        'cpu_compute_precision':'FP32' if recipe.get('weight_storage') else 'See recipe',
        'training_data_count':len(recipe.get('recovery',{}).get('train_indices',[])),
        'evaluation_count':accuracy['examples'],'accuracy_scope':accuracy['evaluation_scope']}
    # Saving the identical snapshot again verifies it and preserves its original date.
    if (bundle/'manifest.json').exists():
        old=read(bundle/'manifest.json')
        manifest['saved_utc']=old['saved_utc']
    (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    document=f'''# Saved winner: {name}

{reason}

- **Inference:** `{manifest['inference_model']}` contains the evaluated model, including stored weights.
- **Further training:** `recovered_state.pt` contains the full recovered PyTorch state (FP32 CPU tensors, including unchanged backbone and trained classifier).
- **Reproduction:** `recipe.json` includes resolution, preprocessing contract, recovery settings and original checkpoint hash. Restore it with `edgeprep.challenge.restore_recipe`; obtain the original supplied checkpoint under `Models/` first.
- **Evidence:** `metrics.json` records full-data accuracy and the hash-matched CPU benchmark; `manifest.json` records checksums and selection reasoning.

Accuracy: {accuracy['accuracy']*100:.4f}% on 27,000 training-pool images; worst class {accuracy['worst_class_accuracy']*100:.2f}%. CPU median latency {benchmark['median_ms']:.4f} ms, {benchmark['threads']} threads. These retain the original measurement protocol, not a new benchmark. Energy is an estimate. FP16 storage is restored to FP32 for CPU compute.

For the recovered model, load with:

```python
from edgeprep.challenge import restore_recipe
model, recipe = restore_recipe("best_models/{bundle.name}/recipe.json")
```

Run the saved ONNX with CPUExecutionProvider. Preserve this version when promoting a future winner.
'''
    (bundle/'README.md').write_text(document,encoding='utf-8')
    current={'name':name,'bundle':str(bundle.relative_to(ROOT)).replace('\\','/'),
        'model_sha256':model_hash,'saved_utc':manifest['saved_utc'],'selection_reason':reason}
    registry=read(base/'registry.json') if (base/'registry.json').exists() else {'winners':[]}
    if not any(r['bundle']==current['bundle'] for r in registry['winners']):registry['winners'].append(current)
    (base/'registry.json').write_text(json.dumps(registry,indent=2),encoding='utf-8')
    temporary=base/'current.tmp.json';temporary.write_text(json.dumps(current,indent=2),encoding='utf-8');temporary.replace(base/'current.json')
    index='# Saved best models\n\n'
    index+=f'**Current selection:** [{name}]({bundle.name}/README.md). `current.json` points to the current winner; `registry.json` preserves the history.\n\n'
    index+='Every version includes inference weights, recovery weights when trained, recipe, checksums and measured results. Older winners remain available.\n\n'
    index+='| Winner | Saved bundle |\n|---|---|\n'
    for winner in registry['winners']:
        folder=Path(winner['bundle']).name
        index+=f'| {winner["name"]} | [Files and results]({folder}/README.md) |\n'
    index+='\n[Future winner workflow](../docs/model_preservation.md).\n'
    (base/'README.md').write_text(index,encoding='utf-8')
    print(json.dumps(current,indent=2))
    return bundle

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True);p.add_argument('--accuracy',required=True)
    p.add_argument('--benchmark',required=True);p.add_argument('--reason',required=True)
    a=p.parse_args();save(a.name,a.accuracy,a.benchmark,a.reason)

if __name__=='__main__':main()
